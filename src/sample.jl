using AbstractMCMC: AbstractMCMC
using AdvancedHMC: AdvancedHMC
using FlexiChains: FlexiChains, SymChain

"""
    SymChain

Re-exported from FlexiChains.jl — the default `chain_type` returned by
[`sample`](@ref). A `Symbol`-keyed `FlexiChains.FlexiChain`, indexable by
`chn[:paramname]`; see FlexiChains.jl's own documentation for the full chain
API (`FlexiChains.rhat`, `chainscat`/`chainsstack` for multi-chain combination,
etc.). PracticalBayes only re-exports the name for convenience — it defines no
behavior of its own on top of it.
"""
SymChain

"""
    sample(rng, model::Model, spl::AdvancedHMC.AbstractHMCSampler, N;
           adtype=AutoForwardDiff(), init=NamedTuple(), chain_type=SymChain,
           kwargs...)

Run `spl` (e.g. `AdvancedHMC.NUTS(0.8)`) on `model` for `N` iterations and
return the posterior draws.

By default the result is a `FlexiChains.SymChain`: every constrained parameter
name in `model` is a `Parameter(:name)` key (mapped back to constrained space
from the sampler's unconstrained draws), and every AdvancedHMC per-iteration
statistic (`log_density`, `acceptance_rate`, `n_steps`, ...) is an
`Extra(:stat_name)` key. Index and summarize the result as usual: `chn[:mu]`,
`chn[Extra(:n_steps)]`, `summarystats(chn)`.

`n_adapts` (default `min(N ÷ 10, 1000)`, AdvancedHMC's own default) sets the
number of step-size/metric adaptation iterations. `discard_initial` (default 0)
runs and drops that many iterations before the `N` kept ones, and `thinning`
(default 1) keeps every `thinning`-th iteration after that — the standard
AbstractMCMC meanings. To drop warm-up, pass `discard_initial=n_adapts`.

The parameter vector uses the sampler's float type
(`AdvancedHMC.sampler_eltype(spl)`), so `NUTS(0.8f0)` samples in Float32 and
`NUTS(0.8)` in Float64.

`init` supplies starting values for named parameters (defaults to a draw from
the prior). Pass `chain_type=nothing` to get AdvancedHMC's raw
`Vector{AdvancedHMC.Transition}` instead — the unconstrained θ vectors, without
the constrained-space mapping.

`progress` shows a live progress meter, defaulting to
[`default_progress()`](@ref) (on when interactive, off in scripts and CI).

Multiple chains use the standard AbstractMCMC interface with no extra setup:

    sample(rng, model, spl, MCMCThreads(), N, nchains; kwargs...)

runs `nchains` independent chains and combines them into a single `SymChain`
with a chain dimension.
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
    layout, θ0, store0 = build_layout(model; init=init, T=T)
    ldf = LogDensityFunction(model, layout, store0, adtype; θ0=θ0)
    ldm = AbstractMCMC.LogDensityModel(ldf)

    # AbstractMCMC's own sampling loop already knows how to draw a progress bar,
    # so hand `progress` straight to it rather than wrapping a second one around.
    if chain_type === nothing
        return AbstractMCMC.sample(rng, ldm, spl, N; initial_params=θ0, progress=progress, kwargs...)
    end

    raw_transitions, state = _pb_sample_transitions(rng, ldm, spl, N; initial_params=θ0,
                                                    progress=progress, kwargs...)
    nt_transitions = [_pb_transition_to_nt(layout, t) for t in raw_transitions]
    return AbstractMCMC.bundle_samples(nt_transitions, ldm, spl, state, chain_type)
end

"""
    sample(rng, model::Model, spl::Gibbs, N;
           init=NamedTuple(), adtype=AutoForwardDiff(), n_adapts=0,
           discard_initial=0, save_states=nothing, chain_type=SymChain, kwargs...)

Run a [`Gibbs`](@ref) sampler on `model` for `N` sweeps and return the draws.

Each sweep updates every block once (NUTS for HMC blocks, `latent_step` for
latent-kernel blocks). By default the result is a `FlexiChains.SymChain` with
one key per model variable. `discard_initial` drops that many leading sweeps
(burn-in); `n_adapts` is passed to every HMC block on every sweep and, as with
AdvancedHMC generally, should be its final target value from the first sweep on
(see [`Gibbs`](@ref)'s step docstring). `init` supplies starting values for any
named variables.

Pass `chain_type=nothing` to get the raw `Vector` of per-sweep `NamedTuple`
transitions instead of a chain.

`progress` shows a live progress meter, defaulting to
[`default_progress()`](@ref) (on when interactive, off in scripts and CI). It
spans the whole run — the `discard_initial` burn-in sweeps as well as the `N`
kept ones — since both cost the same wall-clock time to wait through.

By default a variable whose value is an array (e.g. a latent state matrix) is
retained per draw as that array. For a large latent that is fine to condition on
but too big to keep every draw of — a discrete trajectory `X` can be megabytes
per sweep, so thousands of sweeps materialise gigabytes in RAM — use
`save_states` to say, per variable, where its per-sweep value should go. The
sampler ALWAYS keeps the current value live for conditioning; `save_states` only
changes the OUTPUT:

    save_states = (X = :buffer,)          # keep live, DROP from the chain
    save_states = (X = ("X.jld2", 100),)  # keep live, append to disk every 100 sweeps
    save_states = (X = :chain,)           # keep in the chain (the default)

Values may be `:chain`/`true` (default; retain in the chain), `:buffer`/`false`
(omit from the chain, write nothing), or `(path, every)` (stream to disk every
`every` sweeps, plus a final flush; omitted from the chain). Any variable not
named keeps the default `:chain` behaviour, so scalar parameters are unaffected.
The disk form needs a backend loaded (`using JLD2` activates the bundled sink);
see [`write_state_chunk!`](@ref) to plug in another format.
"""
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
# `AbstractMCMC.sample`, it has to supply the three things that wrapper would:
#
#   * `n_adapts`: `step` itself defaults to 0, i.e. NO step-size or mass-matrix
#     adaptation at all. The default here is AdvancedHMC's own `sample` default,
#     so `chain_type=nothing` (which does go through AdvancedHMC's `sample`) and
#     the chain path adapt identically.
#   * `discard_initial` / `thinning`: implemented by `AbstractMCMC.mcmcsample`,
#     not by `step`, so without this they were silently swallowed and the chain
#     always held every iteration, warm-up included. The step counting below is
#     mcmcsample's: `1 + discard_initial + thinning * (N - 1)` steps in total,
#     keeping the last of the discarded run and then every `thinning`-th step.
function _pb_sample_transitions(rng, ldm, spl, N; initial_params, progress=false,
                                n_adapts::Int=min(div(N, 10), 1_000),
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
        t, state = AbstractMCMC.step(rng, ldm, spl; initial_params=initial_params, n_adapts=n_adapts, kwargs...)
        _progress_update!(p, done)
        for _ in 1:discard_initial
            t, state = AbstractMCMC.step(rng, ldm, spl, state; n_adapts=n_adapts, kwargs...)
            _progress_update!(p, done += 1)
        end
        transitions[1] = t
        for i in 2:N
            for _ in 1:thinning
                t, state = AbstractMCMC.step(rng, ldm, spl, state; n_adapts=n_adapts, kwargs...)
                _progress_update!(p, done += 1)
            end
            transitions[i] = t
        end
        return transitions, state
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

function _pb_transition_to_nt(layout::Layout, t::AdvancedHMC.Transition)
    params = invlink(layout, t.z.θ)
    stats = AdvancedHMC.stat(t)
    return PBTransitionNT(params, stats)
end
