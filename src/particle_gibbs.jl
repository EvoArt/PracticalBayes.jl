# Particle Gibbs: a generic Gibbs kernel for discrete (or any non-differentiable)
# Gibbs blocks, via conditional sequential Monte Carlo (CSMC; Andrieu, Doucet &
# Holenstein 2010). It is what Turing's `PG` is, and it lets a model whose
# latent block has no hand-written FFBS/conjugate kernel still be sampled.
#
# HOW A PARTICLE IS RUN WITHOUT COPYING A RUNNING MODEL
# -----------------------------------------------------
# A particle filter advances every particle to the next observation, reweights,
# resamples (copying some particles, dropping others), and continues. Turing does
# this by pausing the model mid-run and copying the paused task (Libtask). PB
# instead represents a particle by the list of block values it has drawn so far
# (its `trace`), and advances it by RE-RUNNING the model from the start under
# `ParticleMode`:
#
#   * a block site replays the next value from the trace, or, once the trace is
#     exhausted, draws a fresh value from its prior and appends it;
#   * every other site (observes, other blocks' variables, `@addlogprob!`) adds
#     its log density to the accumulator, read from the current Gibbs state;
#   * the run stops (by throwing `_PGStop`) right after the `j`-th WEIGHTING
#     site, i.e. any non-block `~`/`.~` site.
#
# Copying a particle is then copying a short vector, and nothing depends on
# Libtask. The price is that stage `j` re-runs the model prefix up to site `j`,
# so a sweep costs O(J^2) site visits per particle for J weighting sites, where
# a pausing implementation costs O(J). For the models PG is the right tool for
# (tens to a few hundred observations) that is cheap next to the resampling;
# for long series an exact kernel (FFBS, see gibbs_kernel.jl) is far better anyway.
#
# Replaying by visit ORDER is exact: a particle's earlier values are replayed,
# so its control flow up to the stop point is the same as when they were drawn.
#
# WEIGHTS. Block sites are proposed from their prior, so their prior density
# cancels against the proposal and adds nothing; everything else's log density
# is the incremental weight. That is read off the accumulator's running
# `logjoint` total at the stop point, minus the total at the particle's
# previous stop, so `@addlogprob!` terms (which bypass the tilde methods) are
# counted too.

"""
    PG(n_particles)

A Gibbs kernel that draws a [`Gibbs`](@ref) block by particle Gibbs
(conditional sequential Monte Carlo) with `n_particles` particles, at least 2.
Use it for discrete latent variables that have no exact kernel.

```julia
spl = Gibbs(:k => PG(50), (:μ, :w) => NUTS())
chn = sample(model, spl, 1000)
```

More particles mix faster per sweep. Each sweep re-runs the model once per
particle per observation, so the cost grows with the square of the number of
observe statements. For hidden Markov models and similar chains an exact
[`gibbs_step`](@ref) kernel is faster and mixes better.
"""
struct PG <: AbstractGibbsKernel
    n_particles::Int
    function PG(n_particles::Integer)
        n_particles >= 2 || throw(ArgumentError(
            "PG needs at least 2 particles (one is the conditioning reference trajectory), got $n_particles"))
        return new(Int(n_particles))
    end
end

# Thrown to stop a particle's run at its stage boundary. One shared instance:
# nothing about a stop needs to be carried in the exception itself.
struct _PGStop <: Exception end
const _PG_STOP = _PGStop()

"""
    ParticleMode

Evaluation mode used by [`PG`](@ref): replays (or extends) one particle's
block values and accumulates every other site's log density, stopping after a
set number of weighting sites. `names` is the tuple of block variable names, a
type parameter so membership folds to a constant at each site.
"""
mutable struct ParticleMode{names,R<:AbstractRNG,V<:NamedTuple,C<:NamedTuple} <: AbstractEvalMode
    rng::R
    values::V                       # current Gibbs state; non-block sites read from here
    conditioned::C
    trace::Vector{Any}              # the running particle's block values, in visit order
    where_::Vector{Tuple{Symbol,Int}}  # (name, linear index; 0 for a whole-variable site) per trace entry
    pos::Int                        # next trace entry to replay
    from_values::Bool               # reference run: take block values from `values`, not the prior
    stop_at::Int                    # stop after this many weighting sites
    n_weighted::Int                 # weighting sites seen so far in this run
    logw::Float64                   # accumulated log density at the stop (or at the end)
end

# One particle: its block draws so far, how many of them the last run consumed
# (only that prefix is valid to copy), and the log density accumulated up to
# its last stop, which the next stage's increment is measured from.
mutable struct _Particle
    trace::Vector{Any}
    where_::Vector{Tuple{Symbol,Int}}
    used::Int
    W::Float64
end
_Particle() = _Particle(Any[], Tuple{Symbol,Int}[], 0, 0.0)
_copy_prefix(p::_Particle) = _Particle(p.trace[1:p.used], p.where_[1:p.used], p.used, p.W)

@inline _in_block(::ParticleMode{names}, ::Val{s}) where {names,s} = s in names

# A block site's value: replayed if this particle already has it, otherwise
# drawn (from the prior, or from the current state on the reference run) and
# appended to the trace.
function _pg_draw!(m::ParticleMode, s::Symbol, k::Int, dist)
    if m.pos <= length(m.trace)
        x = m.trace[m.pos]
    else
        x = if m.from_values
            v = getfield(m.values, s)
            k == 0 ? v : v[k]
        else
            rand(m.rng, dist)
        end
        push!(m.trace, x)
        push!(m.where_, (s, k))
    end
    m.pos += 1
    return x
end

# Called after every weighting site has been added to `acc`.
@inline function _pg_weighted!(m::ParticleMode, acc::Accum)
    m.n_weighted += 1
    if m.n_weighted >= m.stop_at
        m.logw = logjoint(acc)
        throw(_PG_STOP)
    end
    return acc
end

# --- tilde methods ------------------------------------------------------------

function tilde(m::ParticleMode, vs::Val{s}, dist, value, acc::Accum) where {s}
    # A `missing` argument marks a parameter passed positionally, as for scalar
    # `~` elsewhere; it is an assume, not an observe.
    value === missing && return tilde(m, vs, dist, nothing, acc)
    return value, _pg_weighted!(m, acc_lik(acc, logpdf(dist, value)))
end

function tilde(m::ParticleMode, vs::Val{s}, dist, ::Nothing, acc::Accum) where {s}
    _in_block(m, vs) && return _pg_draw!(m, s, 0, dist), acc
    x = getfield(m.values, s)
    return x, _pg_weighted!(m, acc_prior(acc, logpdf(dist, x)))
end

function tilde_index(m::ParticleMode, vs::Val{s}, container, idx::Tuple, dist, acc::Accum) where {s}
    k = LinearIndices(size(container))[idx...]
    if _in_block(m, vs)
        x = _pg_draw!(m, s, k, dist)
        container[k] = x
        return x, acc
    end
    # Another block's indexed family: its whole array is in the Gibbs state,
    # in linear-index order (as `_assume_index(::ValueSlot, ...)` reads it).
    x = getfield(m.values, s)[k]
    container[k] = x
    return x, _pg_weighted!(m, acc_prior(acc, logpdf(dist, x)))
end

function tilde_index_observe(m::ParticleMode, ::Val{s}, y, idx::Tuple, dist, acc::Accum) where {s}
    v = _data_element(Val(s), y, idx)
    v === missing && _missing_element_error(s, idx)
    return _pg_weighted!(m, acc_lik(acc, logpdf(dist, v)))
end

function tilde_dot(m::ParticleMode, ::Val{s}, y, dist_bcast, acc::Accum) where {s}
    y === nothing && error("`.~` assume (unknown LHS) is not supported; use `x[i] ~ dist` or an array distribution")
    _dot_all_missing(y) && throw(ArgumentError("observe site `$s` has no data; PG cannot weight particles against it"))
    return _pg_weighted!(m, acc_lik(acc, _dot_loglik(dist_bcast, y)))
end

# --- the kernel ---------------------------------------------------------------

# Run `p` from the start until its `stop_at`-th weighting site (or the end of
# the model). Updates `p.used` and returns the accumulated log density.
function _pg_run!(m::ParticleMode, model::Model, p::_Particle, stop_at::Int)
    m.trace = p.trace
    m.where_ = p.where_
    m.pos = 1
    m.n_weighted = 0
    m.stop_at = stop_at
    try
        _, acc = evaluate(model, m, Accum(0.0))
        m.logw = logjoint(acc)
    catch e
        e isa _PGStop || rethrow()
    end
    p.used = m.pos - 1
    return m.logw
end

# Multinomial resampling of particles 2:P; particle 1 is the reference
# trajectory and is always its own ancestor, which is what makes this
# CONDITIONAL SMC (and the resulting Gibbs update exact).
function _pg_ancestors(rng, logw)
    P = length(logw)
    cw = cumsum(_pg_weights(logw))
    a = Vector{Int}(undef, P)
    a[1] = 1
    for i in 2:P
        a[i] = min(searchsortedfirst(cw, rand(rng) * cw[end]), P)
    end
    return a
end

function _pg_select(rng, logw)
    cw = cumsum(_pg_weights(logw))
    return min(searchsortedfirst(cw, rand(rng) * cw[end]), length(cw))
end

# Unnormalised weights, scaled so the largest is 1. A NaN log weight (a
# `logpdf` that produced NaN) counts as impossible, like -Inf. The reference
# particle is the current state of a valid chain, so in practice some weight
# is finite; if none is, every particle is equally (im)possible and the
# weights are left uniform rather than turning into NaNs.
function _pg_weights(logw)
    lw = map(x -> isnan(x) ? -Inf : x, logw)
    mx = maximum(lw)
    isfinite(mx) || return ones(length(lw))
    return exp.(lw .- mx)
end

# The block's new value, as a NamedTuple keyed by `block_names`. Indexed
# families start from a copy of their current array (every element is
# overwritten, since every visited element is in the trace) so the result has
# the same shape and element type the rest of the Gibbs state expects.
function _pg_assemble(p::_Particle, block_names, values::NamedTuple)
    vals = map(block_names) do s
        out = getfield(values, s)
        copied = false
        for i in 1:p.used
            name, k = p.where_[i]
            name === s || continue
            if k == 0
                out = p.trace[i]
            else
                copied || (out = copy(out); copied = true)
                out[k] = p.trace[i]
            end
        end
        out
    end
    return NamedTuple{block_names}(vals)
end

function gibbs_step(rng, k::PG, block_names, c::ModelConditional)
    model = c.model
    P = k.n_particles
    m = ParticleMode{block_names,typeof(rng),typeof(c.values),typeof(model.conditioned)}(
        rng, c.values, model.conditioned, Any[], Tuple{Symbol,Int}[], 1, true, typemax(Int), 0, 0.0)

    # The reference particle is the block's CURRENT value. One full run records
    # it in visit order and counts the weighting sites, which fixes the number
    # of SMC stages J.
    ref = _Particle()
    _pg_run!(m, model, ref, typemax(Int))
    J = m.n_weighted
    ref.used = 0
    m.from_values = false

    particles = [i == 1 ? ref : _Particle() for i in 1:P]
    logw = zeros(P)
    for j in 1:J
        for (i, p) in enumerate(particles)
            W = _pg_run!(m, model, p, j)
            logw[i] += W - p.W
            p.W = W
        end
        j == J && break   # the final weights choose the draw; no resample after them
        a = _pg_ancestors(rng, logw)
        particles = [i == 1 ? particles[1] : _copy_prefix(particles[a[i]]) for i in 1:P]
        fill!(logw, 0.0)
    end

    # Run every particle to the end: draws any block sites after the last
    # weighting site (from the prior, unweighted) and adds any trailing
    # `@addlogprob!` terms.
    for (i, p) in enumerate(particles)
        W = _pg_run!(m, model, p, typemax(Int))
        logw[i] += W - p.W
        p.W = W
    end
    return _pg_assemble(particles[_pg_select(rng, logw)], block_names, c.values)
end
