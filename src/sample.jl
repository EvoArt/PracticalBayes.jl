using AbstractMCMC: AbstractMCMC
using AdvancedHMC: AdvancedHMC
using FlexiChains: FlexiChains, SymChain

"""
    SymChain

The chain type [`sample`](@ref) returns, re-exported from
[FlexiChains.jl](https://github.com/penelopeysm/FlexiChains.jl). Index it by
parameter name: `chn[:μ]` is a `(draws × chains)` matrix. A vector-valued
parameter is one entry holding whole vectors; see [`param_draws`](@ref).
"""
SymChain

"""
    sample(rng, model::Model, spl::AdvancedHMC.AbstractHMCSampler, N;
           adtype=AutoForwardDiff(), init=NamedTuple(), chain_type=SymChain,
           kwargs...)

Draw `N` samples from the posterior of `model` with an HMC sampler such as
`NUTS()` and return them as a [`SymChain`](@ref).

# Keywords
- `n_adapts`: number of warm-up iterations that adapt the step size and
  metric; defaults to `min(1000, N ÷ 2)`.
- `discard_initial`: iterations to run and drop before the `N` kept ones;
  defaults to `n_adapts`, so the chain holds `N` draws after warm-up. Pass
  `discard_initial=0` to keep the warm-up.
- `thinning`: keep every `thinning`-th iteration.
- `init`: starting values by name; others are drawn from the prior.
- `adtype`: AD backend for the gradient, e.g. `AutoMooncake()`.
- `progress`: show a progress meter; defaults to [`default_progress`](@ref).
- `chain_type=nothing`: return AdvancedHMC's raw transitions instead.

Parameters are reported on their constrained scale. Sampler statistics such as
`:acceptance_rate` and `:n_steps`, and the `:logprior`, `:loglikelihood` and
`:logjoint` of each draw, are included as extras. The parameters take the
sampler's number type, so `NUTS(0.8f0)` samples in Float32 (see
[Run a model in Float32](@ref)).

For several chains: `sample(model, spl, MCMCThreads(), N, nchains)`.

```julia
chn = sample(model, NUTS(), 1000)
```
"""
function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::Model,
    spl::AdvancedHMC.AbstractHMCSampler,
    N::Integer;
    adtype=ADTypes.AutoForwardDiff(),
    init=NamedTuple(),
    chain_type=SymChain,
    progress=default_progress(),
    n_adapts::Int=min(1_000, div(N, 2)),
    discard_initial::Int=n_adapts,
    kwargs...,
)
    # Build θ0 in the SAMPLER's float type, not a hardcoded Float64. AdvancedHMC
    # converts `initial_params` to `sampler_eltype(spl)` itself (its
    # `make_initial_params`), so with `NUTS(0.8f0)` it would hand a Vector{Float32}
    # to a gradient prepared against Vector{Float64} and DifferentiationInterface
    # throws `PreparationMismatchError`. Taking the type from the sampler keeps the
    # prepared θ and the sampled θ identical; for the usual `NUTS(0.8)` this is
    # Float64, i.e. exactly what was built before.
    T = AdvancedHMC.sampler_eltype(spl)
    # `rng=rng`: the starting point is drawn from the prior, and it must come
    # from the caller's RNG or `sample(StableRNG(1), ...)` is not reproducible
    # (it used the global RNG until 2026-11-01).
    layout, θ0, store0 = build_layout(model; init=init, T=T, rng=rng)
    ldf = LogDensityFunction(model, layout, store0, adtype; θ0=θ0)
    ldm = AbstractMCMC.LogDensityModel(ldf)

    # `n_adapts` travels on to every `step` call through `kwargs`; see
    # `_pb_sample_transitions` for why it must be supplied explicitly here.
    #
    # `chain_type=nothing` runs the SAME loop and keeps the raw transitions.
    # It used to hand off to AdvancedHMC's own `sample`, whose progress meter
    # (`HMCProgressCallback`) only accepts an `HMCState` — so PB's own HMC
    # samplers with their own state type (`NUTSthenHMC`) threw a MethodError
    # there whenever progress was on. One loop for both outputs also means the
    # raw and chain paths cannot disagree on warm-up, discard or thinning.
    record = chain_type === nothing ? (t, _) -> t : (t, _) -> _pb_transition_to_nt(model, layout, t)
    transitions, state = _pb_sample_transitions(rng, ldm, spl, N, record;
                                                initial_params=θ0, progress=progress,
                                                n_adapts=n_adapts, discard_initial=discard_initial, kwargs...)
    chain_type === nothing && return transitions
    return AbstractMCMC.bundle_samples(transitions, ldm, spl, state, chain_type)
end

# `NUTS(; adtype=...)` (nuts.jl): the sampler's own backend becomes `sample`'s
# `adtype`. An explicit `adtype=` keyword to `sample` is overridden by it, as
# the sampler's is the more specific request.
function AbstractMCMC.sample(rng::Random.AbstractRNG, model::Model, spl::_ADSampler, N::Integer; kwargs...)
    return AbstractMCMC.sample(rng, model, spl.sampler, N; kwargs..., adtype=spl.adtype)
end

"""
    sample(rng, model::Model, spl::AbstractMCMC.AbstractSampler, N;
           adtype=AutoForwardDiff(), init=NamedTuple(), chain_type=SymChain,
           kwargs...)

Draw `N` samples with any AbstractMCMC sampler that works on a log density and
implements `AbstractMCMC.getparams`, such as AdvancedMH's random-walk sampler
or a slice sampler. Returns a [`SymChain`](@ref).

The sampler works on the unconstrained parameter vector, with a gradient from
`adtype` for samplers that use one. `discard_initial` and `thinning` have their
AbstractMCMC meanings; nothing is discarded by default. Every parameter must be
continuous; for discrete ones use [`Gibbs`](@ref) with a latent kernel.
"""
function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::Model,
    spl::AbstractMCMC.AbstractSampler,
    N::Integer;
    adtype=ADTypes.AutoForwardDiff(),
    init=NamedTuple(),
    chain_type=SymChain,
    progress=default_progress(),
    kwargs...,
)
    layout, θ0, store0 = build_layout(model; init=init, rng=rng)
    ldf = LogDensityFunction(model, layout, store0, adtype; θ0=θ0)
    ldm = AbstractMCMC.LogDensityModel(ldf)

    if chain_type === nothing
        return AbstractMCMC.sample(rng, ldm, spl, N; initial_params=θ0, progress=progress, kwargs...)
    end

    # Read each kept draw off the sampler's STATE rather than its transition:
    # `getparams(model, state)` is the interface every compliant sampler
    # implements, whereas transition types differ package to package.
    record = function (_, state)
        params = invlink(layout, AbstractMCMC.getparams(ldm, state))
        return PBTransitionNT(params, merge(_sampler_stats(state), _density_stats(model, params)))
    end
    transitions, state = _pb_sample_transitions(rng, ldm, spl, N, record;
                                                initial_params=θ0, progress=progress, kwargs...)
    return AbstractMCMC.bundle_samples(transitions, ldm, spl, state, chain_type)
end

# `getstats` is optional in AbstractMCMC's interface: a sampler that does not
# implement it simply contributes no `Extra` keys to the chain.
_sampler_stats(state) = applicable(AbstractMCMC.getstats, state) ? AbstractMCMC.getstats(state) : NamedTuple()

# Project a full sweep NamedTuple onto a statically-known subset of its keys.
# `Val{names}` makes `names` a compile-time constant, so the returned NamedTuple's
# type is inferred and no runtime key search happens — the per-sweep cost is a
# handful of field loads, and (crucially) it drops references to the omitted
# large arrays so they can be freed instead of accumulating in the chain.
@inline function _select_names(nt::NamedTuple, ::Val{names}) where {names}
    NamedTuple{names}(nt)
end

# The disk-streaming bookkeeping: one growable buffer per streamed variable,
# flushed to its OWN file every `every` sweeps. Kept out of the sweep loop's type
# domain (a plain Vector of little structs) — the loop just pushes and, on the
# flush boundary, hands whole chunks off. `Any`-typed buffers are fine here: this
# is once-per-sweep bookkeeping on already-materialised values, not the hot path,
# and the values are heterogeneous arrays we only ever move, never compute on.
# `n_written` counts sweeps already flushed to disk, so the next flush knows its
# `iters_x_to_y` range.
mutable struct _DiskStream
    name::Symbol
    disp::SaveToDisk
    buffer::Vector{Any}
    n_written::Int
end

function _init_disk_streams(save_states::NamedTuple)
    streams = _DiskStream[]
    for name in keys(save_states)
        disp = save_states[name]
        disp isa SaveToDisk && push!(streams, _DiskStream(name, disp, Vector{Any}(), 0))
    end
    streams
end

# Append one sweep's value for each streamed variable; flush any buffer that has
# reached its interval. `_flush_disk!` forces a final flush of trailing partials.
function _record_disk!(streams::Vector{_DiskStream}, t::NamedTuple)
    @inbounds for s in streams
        push!(s.buffer, getfield(t, s.name))
        length(s.buffer) >= s.disp.every && _flush_stream!(s)
    end
end
function _flush_disk!(streams::Vector{_DiskStream})
    for s in streams
        isempty(s.buffer) || _flush_stream!(s)
    end
end
function _flush_stream!(s::_DiskStream)
    first_iter = s.n_written + 1
    last_iter = s.n_written + length(s.buffer)
    write_state_chunk!(s.disp, s.name, s.buffer, first_iter, last_iter)
    s.n_written = last_iter
    empty!(s.buffer)
    nothing
end

"""
    sample(rng, model::Model, spl::Gibbs, N; n_adapts=0, discard_initial=0,
           init=NamedTuple(), adtype=AutoForwardDiff(), save_states=nothing,
           progress=default_progress(), chain_type=SymChain)

Run `N` sweeps of the [`Gibbs`](@ref) sampler `spl` after `discard_initial`
sweeps of burn-in, and return them as a [`SymChain`](@ref).

# Keywords
- `n_adapts`: warm-up iterations for each HMC block, counted from the first
  sweep.
- `discard_initial`: sweeps to run and drop before the `N` kept ones.
- `init`: starting values by name; others are drawn from the prior.
- `adtype`: AD backend used by every HMC block.
- `save_states`: where each variable's draws go, by name. `:chain` (the
  default) keeps them in the chain; `:buffer` leaves them out; `(path, every)`
  writes them to disk every `every` sweeps (needs `using JLD2`; read them back
  with [`read_states`](@ref)).
- `chain_type=nothing`: return a vector of `NamedTuple`s instead.

```julia
chn = sample(model, spl, 1000; n_adapts=500, discard_initial=500,
             save_states=(z = :buffer,))
```
"""
function AbstractMCMC.sample(
    rng::Random.AbstractRNG,
    model::Model,
    spl::Gibbs,
    N::Integer;
    init=NamedTuple(),
    adtype=ADTypes.AutoForwardDiff(),
    n_adapts::Int=0,
    discard_initial::Int=0,
    save_states=nothing,
    chain_type=SymChain,
    progress=default_progress(),
    kwargs...,
)
    save = _normalize_save_states(save_states)
    streams = _init_disk_streams(save)

    # The meter spans burn-in AND kept sweeps: `discard_initial` sweeps cost the
    # same as kept ones, so a bar covering only `N` would sit at 0% through the
    # entire warm-up. `done` counts sweeps actually taken, the first one included.
    total_sweeps = 1 + discard_initial + (discard_initial == 0 ? N - 1 : N)
    transitions, state = _run_with_progress(progress, "Sampling", total_sweeps) do p
        done = 0

        # First sweep builds the per-block layouts/preps and takes one step; every
        # later sweep reuses them. We keep `N` sweeps after discarding the first
        # `discard_initial` as burn-in.
        t, state = AbstractMCMC.step(rng, model, spl; init=init, adtype=adtype, n_adapts=n_adapts, kwargs...)
        _progress_update!(p, done += 1)
        # A `save_states` name that isn't a real model variable is almost certainly a
        # typo — catch it here (once, cheaply) rather than silently doing nothing.
        for name in keys(save)
            haskey(t, name) || throw(ArgumentError(
                "`save_states` names variable `$name`, which is not a variable of this model " *
                "(its variables are: $(join(keys(t), ", ")))"))
        end
        for _ in 1:discard_initial
            t, state = AbstractMCMC.step(rng, model, spl, state; n_adapts=n_adapts, kwargs...)
            _progress_update!(p, done += 1)
        end

        # Which of the sweep's variables survive into the chain, as a compile-time
        # constant tuple (the sweep always returns the same key set, so this is fixed
        # for the whole run). Everything else is either dropped (`:buffer`) or
        # streamed to disk — in both cases it must NOT be retained here, which is the
        # whole point: retaining a 3 MB `X` for 5000 sweeps is what exhausts memory.
        retained = Val(_retained_keys(t, save))

        kept = _select_names(t, retained)
        transitions = Vector{typeof(kept)}(undef, N)
        # When there is no burn-in, the very first sweep (`t` above) is the first
        # kept draw; otherwise `t` currently holds the last discarded sweep and the
        # first kept draw comes from the next step.
        if discard_initial == 0
            transitions[1] = kept
            _record_disk!(streams, t)
            for i in 2:N
                t, state = AbstractMCMC.step(rng, model, spl, state; n_adapts=n_adapts, kwargs...)
                transitions[i] = _select_names(t, retained)
                _record_disk!(streams, t)
                _progress_update!(p, done += 1)
            end
        else
            for i in 1:N
                t, state = AbstractMCMC.step(rng, model, spl, state; n_adapts=n_adapts, kwargs...)
                transitions[i] = _select_names(t, retained)
                _record_disk!(streams, t)
                _progress_update!(p, done += 1)
            end
        end
        _flush_disk!(streams)  # trailing partial buffers

        return transitions, state
    end

    chain_type === nothing && return transitions
    # A Gibbs transition is already a constrained-parameter NamedTuple, so
    # FlexiChains' generic `to_nt_and_stats(::NamedTuple)` bundles it directly
    # (no params/stats split to preserve, unlike the AdvancedHMC path below).
    return AbstractMCMC.bundle_samples(transitions, model, spl, state, chain_type)
end

# The chain-retained key subset of a full sweep NamedTuple, preserving the
# sweep's own key order. Runs ONCE per `sample` call (not per sweep), so the
# `keys`/`haskey` work here is immaterial.
function _retained_keys(t::NamedTuple, save::NamedTuple)
    Tuple(k for k in keys(t) if _retained_in_chain(save, k))
end

# `bundle_samples` receives whatever `AbstractMCMC.step` returns as a
# "transition" — for AdvancedHMC that's `AdvancedHMC.Transition` (a `PhasePoint`
# plus a stats `NamedTuple`), not a plain `NamedTuple`. We collect the raw
# transitions ourselves and convert each to `(constrained_params, stats)` before
# handing off to `bundle_samples`, so FlexiChains' generic `NamedTuple` path does
# the chain-building rather than a bespoke `FlexiChain{Symbol}` method here.
#
# Because this loop calls `step` directly rather than going through
# `AbstractMCMC.sample`, it has to supply the things that wrapper would:
#
#   * `n_adapts` (HMC only): AdvancedHMC's `step` defaults it to 0, i.e. NO
#     step-size or mass-matrix adaptation at all. The HMC `sample` method above
#     therefore passes AdvancedHMC's own `sample` default through `kwargs`, so
#     `chain_type=nothing` (which does go through AdvancedHMC's `sample`) and
#     the chain path adapt identically. It is not a keyword here because other
#     samplers have no such argument.
#   * `discard_initial` / `thinning`: implemented by `AbstractMCMC.mcmcsample`,
#     not by `step`, so without this they were silently swallowed and the chain
#     always held every iteration, warm-up included. The step counting below is
#     mcmcsample's: `1 + discard_initial + thinning * (N - 1)` steps in total,
#     keeping the last of the discarded run and then every `thinning`-th step.
#
# `record(transition, state)` turns each KEPT step into what the chain stores.
# Converting as we go means only kept draws are ever held, in their final form.
function _pb_sample_transitions(rng, ldm, spl, N, record; initial_params, progress=false,
                                discard_initial::Int=0, thinning::Int=1, kwargs...)
    N > 0 || throw(ArgumentError("the number of samples must be ≥ 1, got $N"))
    discard_initial >= 0 || throw(ArgumentError("`discard_initial` must be non-negative, got $discard_initial"))
    thinning >= 1 || throw(ArgumentError("`thinning` must be ≥ 1, got $thinning"))
    transitions = Vector{Any}(undef, N)
    total_steps = 1 + discard_initial + thinning * (N - 1)
    # `progress` is consumed here, NOT forwarded: `AbstractMCMC.step` takes no
    # such keyword, and AdvancedHMC would pass it on into its own internals.
    # The meter spans every step taken, discarded ones included, since they
    # cost the same wall-clock time.
    _run_with_progress(progress, "Sampling", total_steps) do p
        done = 1
        t, state = AbstractMCMC.step(rng, ldm, spl; initial_params=initial_params, kwargs...)
        _progress_update!(p, done)
        for _ in 1:discard_initial
            t, state = AbstractMCMC.step(rng, ldm, spl, state; kwargs...)
            _progress_update!(p, done += 1)
        end
        transitions[1] = record(t, state)
        for i in 2:N
            for _ in 1:thinning
                t, state = AbstractMCMC.step(rng, ldm, spl, state; kwargs...)
                _progress_update!(p, done += 1)
            end
            transitions[i] = record(t, state)
        end
        # Every element has the same type, so narrow the `Vector{Any}` before
        # handing it to `bundle_samples`.
        return identity.(transitions), state
    end
end

# Keeps the constrained-parameter NamedTuple and AdvancedHMC's per-transition
# stat NamedTuple as two separate fields. `FlexiChains.to_nt_and_stats` routes
# the first into the chain's parameters and the second into its extras; a merged
# NamedTuple would land entirely in the parameters half, mislabeling every
# sampler diagnostic as a parameter. `to_nt_and_stats` is FlexiChains' extension
# point for a custom sampler's transition type.
struct PBTransitionNT{P<:NamedTuple,S<:NamedTuple}
    params::P
    stats::S
end
FlexiChains.to_nt_and_stats(t::PBTransitionNT) = (t.params, t.stats)

function _pb_transition_to_nt(model, layout::Layout, t::AdvancedHMC.Transition)
    params = invlink(layout, t.z.θ)
    stats = merge(AdvancedHMC.stat(t), _density_stats(model, params))
    return PBTransitionNT(params, stats)
end

# Turing's chains carry `logprior`, `loglikelihood` and `logjoint` for every
# draw, on the constrained scale (no Jacobian term), and Turing code reads them
# (`chn[:logjoint]`). AdvancedHMC's own `log_density` is not that: it is on the
# unconstrained scale, Jacobian included. One FixedMode evaluation per KEPT
# draw gives Turing's three exactly — the same cost Turing pays. A point the
# model cannot score (a `missing` observation, which sampling never needs)
# reports NaN rather than failing the chain.
function _density_stats(model, params::NamedTuple)
    try
        _, acc, _ = _fixed_evaluate(Random.default_rng(), model, params; predict=false)
        return (logprior=Float64(logprior(acc)), loglikelihood=Float64(loglikelihood_(acc)),
                logjoint=Float64(logjoint(acc)))
    catch e
        e isa InterruptException && rethrow()
        return (logprior=NaN, loglikelihood=NaN, logjoint=NaN)
    end
end
