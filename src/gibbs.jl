using AbstractMCMC: AbstractMCMC
using AdvancedHMC: AdvancedHMC
using ADTypes: AutoForwardDiff
import LogDensityProblems

# ===========================================================================
# Block identity: a Symbol or tuple of Symbols, matching `build_layout`'s
# existing `flat=`/`values=` contract exactly (no `AbstractPPL.VarName` here
# — this package only ever conditions on whole top-level names, and every
# other user-facing API already uses plain Symbols for this).
# ===========================================================================

_normalize_block_names(s::Symbol) = (s,)
_normalize_block_names(t::Tuple{Vararg{Symbol}}) = t

"""
    GibbsBlock(names, kernel)

One block of a `Gibbs` sampler: `names` (a `Symbol` or `Tuple` of `Symbol`s)
identifies which model variables this block owns; `kernel` is either an
AbstractMCMC sampler over the block's unconstrained parameters (e.g.
`NUTS(0.8)`, or AdvancedMH's `RWMH`) or an `AbstractGibbsKernel` (see
gibbs_kernel.jl), such as [`PG`](@ref) or a hand-written FFBS kernel.
"""
struct GibbsBlock{K,N<:Tuple{Vararg{Symbol}}}
    names::N
    kernel::K
end

"""
    Gibbs(pairs::Pair...)

A Gibbs sampler that updates blocks of variables in turn, once each per sweep.

Each pair maps a variable name, or a tuple of names, to the sampler for that
block:

- an HMC sampler (`NUTS`, `HMC`, [`AdaptiveHMC`](@ref),
  [`NUTSthenHMC`](@ref)) or another AbstractMCMC sampler that works on a log
  density, which takes one step on the block's continuous parameters with every
  other variable held fixed;
- a Gibbs kernel ([`PG`](@ref), [`ConjugateGibbs`](@ref) or your own
  [`AbstractGibbsKernel`](@ref)), which draws the block's new values directly.

Every parameter of the model must be in exactly one block, and discrete
parameters need a Gibbs kernel. Run it with [`sample`](@ref); pass `n_adapts`
there and keep it the same for the whole run.

```julia
# `pk` and `AllocationKernel` as in the `ConjugateGibbs` and `gibbs_step` examples
spl = Gibbs((:α, :β) => NUTS(), :p => pk, :z => AllocationKernel())
chn = sample(model, spl, 1000; n_adapts=500, discard_initial=500)
```
"""
struct Gibbs{P<:Tuple{Vararg{GibbsBlock}}} <: AbstractMCMC.AbstractSampler
    blocks::P
end
function Gibbs(pairs::Pair...)
    blocks = ntuple(length(pairs)) do i
        names, kernel = pairs[i]
        GibbsBlock(_normalize_block_names(names), kernel)
    end
    return Gibbs(blocks)
end

_is_kernel_block(b::GibbsBlock) = b.kernel isa AbstractGibbsKernel

# ===========================================================================
# Per-block persistent state, carried inside GibbsState across sweeps.
# ===========================================================================

# Sampler block: `layout` (flat = this block's names, values = every other
# block's names) and `prep` (DI.prepare_gradient) are each built once, at
# first-step initialization, and never rebuilt — only `store`'s values change
# sweep to sweep (its field-name set, and therefore the type `prep` was built
# against, is fixed for the lifetime of a Gibbs run), so reusing `prep` across
# sweeps via LogDensityFunction's inner constructor avoids re-preparing the AD
# backend every sweep. See logdensity.jl's inner-constructor docstring for the
# validity condition.
struct GibbsSamplerSub{L<:Layout,AD,P,S}
    layout::L
    adtype::AD
    prep::P
    hmc_state::S  # `nothing` before this block's first step, then an AdvancedHMC.HMCState
end

# Kernel block: no persistent Layout/prep at all — `gibbs_step` (gibbs_kernel.jl)
# reads/writes plain NamedTuple values directly, never builds a
# LogDensityFunction, never touches AD. Any state a kernel itself needs
# (e.g. an adaptive proposal) lives inside the user's own `kernel` struct,
# which Gibbs treats as opaque and never mutates.
struct GibbsKernelSub end

"""
    GibbsState(values, subs)

The state of a [`Gibbs`](@ref) sampler between sweeps, as returned by
`AbstractMCMC.step`. `values` holds the current value of every parameter;
`subs` holds each block's sampler state. Read the HMC tuning from it with
[`hmc_tuning`](@ref).
"""
struct GibbsState{V<:NamedTuple,C<:Tuple}
    values::V
    subs::C
end

# ===========================================================================
# Coverage validation — every assumed name in exactly one block; no
# discrete/latent-role name assigned to an HMC block.
# ===========================================================================

function _validate_gibbs_coverage(model::Model, spl::Gibbs, rng=Random.default_rng())
    tmode = TraceMode(rng, model.conditioned, NamedTuple())
    evaluate(model, tmode, Accum(0.0))
    records = [r for r in tmode.sites]
    assumed = filter(r -> r.role != :observed, records)

    seen = Dict{Symbol,Int}()  # name -> which block index claims it (first one wins for the error message)
    all_names = Set{Symbol}()
    for (i, b) in enumerate(spl.blocks), n in b.names
        push!(all_names, n)
        if haskey(seen, n)
            throw(ArgumentError("variable `$n` is assigned to more than one Gibbs block (blocks $(seen[n]) and $i)"))
        end
        seen[n] = i
    end

    missing_names = Symbol[r.name for r in assumed if !(r.name in all_names)]
    isempty(missing_names) || throw(ArgumentError(
        "the following assumed model variable(s) are not assigned to any Gibbs block: " *
        join(missing_names, ", ") * ". Every assumed variable must appear in exactly one block.",
    ))

    role_by_name = Dict(r.name => r.role for r in assumed)
    for (i, b) in enumerate(spl.blocks), n in b.names
        if !_is_kernel_block(b) && get(role_by_name, n, :param) == :latent
            throw(ArgumentError(
                "block $i assigns discrete/latent variable `$n` to $(typeof(b.kernel)), a sampler over " *
                "unconstrained continuous parameters — discrete variables need an `AbstractGibbsKernel` " *
                "instead (e.g. `PG(n)`, or a hand-written FFBS/conjugate-Gibbs kernel), since there's no " *
                "continuous unconstrained encoding for them.",
            ))
        end
    end
    return records
end

# ===========================================================================
# First-step initialization: draw an initial value for every name (from
# `init` if supplied, else the prior, via one TraceMode pass reused from
# `_validate_gibbs_coverage`), then build each sampler block's Layout+prep.
# ===========================================================================

"""
    _gibbs_init_values(records, init) -> NamedTuple

Draws (or takes from `init`) a starting constrained-space value for every
assumed model variable. `records` is one `SiteRecord` per tilde site VISIT
— for an ordinary scalar `x ~ dist` that's one record, but for an indexed
family (`x[i] ~ dist` inside a loop) it's one record PER INDEX, all sharing
the same `name`. Mirrors `build_layout`'s by-name grouping (layout.jl) so an
indexed family's initial value is the full vector/array (ordered by first-seen
index), not just the last record for that name. Collapsing a family to its last
element would leave a scalar where a whole latent trajectory (e.g. an HMM's `z`)
is expected, and `_assume_index` would then hit a `BoundsError` indexing that
scalar.
"""
function _gibbs_init_values(records, init::NamedTuple)
    seen_order = Symbol[]
    by_name = Dict{Symbol,Vector}()
    for r in records
        r.role == :observed && continue
        if !haskey(by_name, r.name)
            push!(seen_order, r.name)
            # Concretely typed from the very first record's `init_val`
            # (e.g. `Vector{Int}` for a `Categorical`-typed indexed family,
            # `Vector{Float64}` for a continuous one) — NOT `Any[]`. This
            # matters beyond tidiness: `_init_block_sub` builds `store0`
            # (and therefore `DI.prepare_gradient`'s `prep`) from THIS
            # NamedTuple's element types, and `DifferentiationInterface`
            # strictly checks that every later sweep's `store` has the
            # exact same types `prep` was built against — a `Vector{Any}`
            # here vs. a concretely-typed `Vector{Int64}` from a real
            # `gibbs_step` call is a type mismatch that DI rejects at sweep
            # time with a `PreparationMismatchError`.
            by_name[r.name] = [r.init_val]
        else
            push!(by_name[r.name], r.init_val)
        end
    end
    pairs = Pair{Symbol,Any}[]
    for name in seen_order
        vals = by_name[name]
        default = length(vals) == 1 ? vals[1] : vals
        push!(pairs, name => (haskey(init, name) ? getfield(init, name) : default))
    end
    return NamedTuple(pairs)
end

function _init_block_sub(model::Model, block::GibbsBlock, values0::NamedTuple; adtype=AutoForwardDiff())
    if _is_kernel_block(block)
        return GibbsKernelSub()
    end
    # A block whose sampler carries its own backend (`NUTS(; adtype=...)`) uses it.
    adtype = _block_adtype(block.kernel, adtype)
    other_names = Tuple(k for k in keys(values0) if !(k in block.names))
    own_init = NamedTuple{block.names}(Tuple(values0[k] for k in block.names))
    layout, θ0, _ = build_layout(model; flat=block.names, values=other_names, init=own_init)
    store0 = NamedTuple{other_names}(Tuple(values0[k] for k in other_names))
    ldf0 = LogDensityFunction(model, layout, store0, adtype; θ0=θ0)
    # `ldf0.adtype`, not `adtype`: a block holding every parameter has an empty
    # store, so a GLM-shaped model gets the closed-form gradient (GradMode) and
    # its prep. Every sweep rebuilds the density from this pair, and the prep
    # under the requested backend's type would send it down the AD path.
    return GibbsSamplerSub(layout, ldf0.adtype, ldf0.prep, nothing)
end

# ===========================================================================
# AbstractMCMC.step — the two entry points AdvancedHMC's own sampler
# follows: no-state (first call) and with-state (every subsequent call).
# ===========================================================================

"""
    AbstractMCMC.step(rng, model::Model, spl::Gibbs; init=NamedTuple(), adtype=AutoForwardDiff(), kwargs...)

Take the first Gibbs sweep. Checks that every parameter is in exactly one
block, starts from `init` or the prior, and returns `(values, state)`. Every
HMC block uses the same `adtype`.
"""
function AbstractMCMC.step(rng::Random.AbstractRNG, model::Model, spl::Gibbs; init=NamedTuple(), adtype=AutoForwardDiff(), kwargs...)
    records = _validate_gibbs_coverage(model, spl, rng)
    values0 = _gibbs_init_values(records, init)
    subs0 = ntuple(i -> _init_block_sub(model, spl.blocks[i], values0; adtype=adtype), length(spl.blocks))
    state = GibbsState(values0, subs0)
    return AbstractMCMC.step(rng, model, spl, state; kwargs...)
end

"""
    AbstractMCMC.step(rng, model::Model, spl::Gibbs, state::GibbsState; n_adapts=0, kwargs...)

Take one Gibbs sweep: update each block in the order given to [`Gibbs`](@ref).
Returns `(values, state)`.

Pass the same `n_adapts` on every call, starting with the first: the
adaptation schedule is fixed from the value given then.
"""
function AbstractMCMC.step(rng::Random.AbstractRNG, model::Model, spl::Gibbs, state::GibbsState; n_adapts::Int=0, kwargs...)
    values = state.values
    new_subs = Vector{Any}(undef, length(spl.blocks))
    for i in eachindex(spl.blocks)
        block = spl.blocks[i]
        sub = state.subs[i]
        if _is_kernel_block(block)
            c = ModelConditional(model, values)
            newvals = gibbs_step(rng, block.kernel, block.names, c)
            keys(newvals) == block.names || throw(ArgumentError(
                "gibbs_step for block $(block.names) must return a NamedTuple with exactly those keys, got $(keys(newvals))",
            ))
            values = merge(values, newvals)
            new_subs[i] = GibbsKernelSub()
        else
            other_names = Tuple(k for k in keys(values) if !(k in block.names))
            store = NamedTuple{other_names}(Tuple(values[k] for k in other_names))
            own = NamedTuple{block.names}(Tuple(values[k] for k in block.names))
            θ_now = link(sub.layout, own)

            ldf = LogDensityFunction(model, sub.layout, store, sub.adtype, sub.prep, false)
            ldm = AbstractMCMC.LogDensityModel(ldf)
            spl_i = _block_sampler(block.kernel)   # unwraps `NUTS(; adtype=...)`

            if sub.hmc_state === nothing
                # `n_adapts` MUST be passed here, not only on subsequent steps.
                # AdvancedHMC initialises the Stan windowed adaptor from inside
                # `adapt!`, and ONLY on the block's very first HMC step
                # (`sampler.jl`: `i == 1 && Adaptation.initialize!(adaptor, n_adapts)`),
                # which is itself guarded by `adapt = i <= n_adapts`. Defaulting
                # `n_adapts` to 0 here made that guard false on step 1, so
                # `initialize!` never ran, the adaptor kept `window(0, 0)` with no
                # window splits, and the MASS MATRIX WAS NEVER UPDATED — every run
                # sampled with the identity metric. The step size still adapted
                # (dual averaging needs no window), which is what made this silent:
                # chains looked tuned and merely mixed badly.
                # Verified by driving AdvancedHMC's step path directly with and
                # without `n_adapts` on the first step: without it the adaptor
                # reports `window(0, 0)` and `M⁻¹ ≈ I`; with it, `window(76, 150)`
                # and a correctly recovered metric.
                _, hmc_state = AbstractMCMC.step(rng, ldm, spl_i; initial_params=θ_now,
                                                 _adapt_kwargs(spl_i, n_adapts)..., kwargs...)
            else
                # `setparams!!` against the NEW `ldm` is what makes a sampler's
                # cached log density reflect the other blocks' latest values.
                refreshed = AbstractMCMC.setparams!!(ldm, sub.hmc_state, θ_now)
                _, hmc_state = AbstractMCMC.step(rng, ldm, spl_i, refreshed;
                                                 _adapt_kwargs(spl_i, n_adapts)..., kwargs...)
            end
            θ_new = AbstractMCMC.getparams(hmc_state)
            newvals = invlink(sub.layout, θ_new; include_untracked=true)
            values = merge(values, newvals)
            new_subs[i] = GibbsSamplerSub(sub.layout, sub.adtype, sub.prep, hmc_state)
        end
    end
    newstate = GibbsState(values, Tuple(new_subs))
    return values, newstate
end

# `n_adapts` is an AdvancedHMC keyword. Other samplers (AdvancedMH etc.) have
# no adaptation argument of that name, so it is passed to HMC blocks only.
_adapt_kwargs(::AdvancedHMC.AbstractHMCSampler, n_adapts) = (; n_adapts=n_adapts)
_adapt_kwargs(_, _) = (;)
