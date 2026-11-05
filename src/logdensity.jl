using LogDensityProblems: LogDensityProblems
using ADTypes: ADTypes, AbstractADType
import DifferentiationInterface as DI

"""
    LogDensityFunction(model, layout, store=NamedTuple(), adtype=nothing;
                       θ0=zeros(layout.dim), reject_errors=false, closed_form=true)

The log joint density of `model` as a function of the flat unconstrained
vector `θ`, implementing the LogDensityProblems.jl interface. Values in `store`
are held constant.

# Arguments
- `layout`, `store`: from [`build_layout`](@ref).
- `adtype`: an ADTypes backend such as `AutoForwardDiff()` or `AutoMooncake()`,
  [`AutoPBForwardDiff`](@ref), [`GradMode`](@ref), or `nothing` for the
  density only.

# Keywords
- `θ0`: a vector of the type you will evaluate at; the gradient is prepared for
  its element type. Pass a `Float32` vector for a Float32 run.
- `reject_errors`: treat an error inside the model as density `-Inf` instead of
  throwing.
- `closed_form`: use the closed-form gradient when the model is recognised (see
  [`is_gradmode_compatible`](@ref)). Set `false` to force `adtype`. Ignored for
  [`AutoPBForwardDiff`](@ref), which always uses Piste.

Computation follows the element type of `θ`.

```jldoctest
julia> using LogDensityProblems, ADTypes

julia> @model function demo()
           μ ~ Normal(0, 1)
       end;

julia> layout, θ0, store0 = build_layout(demo());

julia> f = LogDensityFunction(demo(), layout, store0, AutoForwardDiff(); θ0=θ0);

julia> LogDensityProblems.logdensity_and_gradient(f, [1.0])
(-1.4189385332046727, [-1.0])
```
"""
struct LogDensityFunction{M<:Model,L<:Layout,S<:NamedTuple,AD,P}
    model::M
    layout::L
    store::S
    adtype::AD
    prep::P
    reject_errors::Bool
end

function LogDensityFunction(
    model, layout, store=NamedTuple(), adtype=nothing; θ0=zeros(Float64, layout.dim),
    reject_errors=false, closed_form=true
)
    if adtype === nothing
        # order-0: density only, e.g. for a plain Metropolis-Hastings kernel
        # that never needs a gradient. `prep` is `nothing` and never touched.
        return LogDensityFunction(model, layout, store, nothing, nothing, reject_errors)
    end
    if adtype isa GradMode
        # Opt-in closed-form gradient (gradmode_codegen.jl). `prep` holds the
        # recognized plan plus its scratch buffers instead of an AD tape.
        plan = gradmode_plan(model.f)
        plan === nothing && throw(ArgumentError(
            "GradMode() was requested but this model was not recognized as a GLM with a " *
            "closed-form gradient, so the fast path cannot be used. Use a general AD " *
            "backend (e.g. AutoForwardDiff()/AutoMooncake()) instead. See " *
            "`recognize_glm` for the accepted model grammar."))
        isempty(store) || throw(ArgumentError(
            "GradMode() does not support a non-empty `store` (Gibbs blocks / latent " *
            "variables): the closed form is derived for the whole model at once. Use a " *
            "general AD backend for this block."))
        N = length(getfield(model.args, plan.response))
        prep = GradModePrep(plan, GradModeWorkspace{eltype(θ0)}(N), gradmode_cache(plan, layout))
        return LogDensityFunction(model, layout, store, adtype, prep, reject_errors)
    end
    # AUTOMATIC CLOSED FORM. If the model was recognized as GLM-shaped at
    # `@model` expansion time, use the analytic gradient instead of
    # differentiating it — for ANY backend, not just an explicit `GradMode()`.
    #
    # This is worth a lot and is not a micro-optimisation. Measured on the
    # normal-regression benchmark, gradient time:
    #
    #     N=1000 K=50    GradMode  76 us   Mooncake  132 us   Piste tape  222 us
    #     N=5000 K=200   GradMode 351 us   Mooncake 3118 us   Piste tape 3862 us
    #
    # i.e. ~9x faster than the fastest general AD at the large end, because it
    # does no taping at all: the GLM gradient is `X' * resid`, one BLAS call.
    # No amount of tuning a tape reaches that, since the tape's mere existence
    # is the cost — profiling put Mooncake's ENTIRE gradient below the cost of
    # our forward taping pass alone.
    #
    # Opt out with `closed_form=false` when you want the AD path specifically
    # (e.g. checking the closed form against it, which `check_gradmode` does).
    #
    # Never for `AutoPBForwardDiff`: that backend exists only so a model can be
    # compiled with `juliac --trim`, and the closed-form path does not survive
    # the trimmer. Substituting it would quietly defeat the one thing the user
    # asked for by naming that backend (seen: the trimming guide had to pass
    # `closed_form=false` to get a GLM-shaped model to trim).
    if closed_form && !(adtype isa GradMode) && !(adtype isa AutoPBForwardDiff)
        gm = _maybe_gradmode(model, layout, store, θ0)
        if !isempty(gm)
            return LogDensityFunction(model, layout, store, GradMode(), gm[1], reject_errors)
        end
    end
    if adtype isa AutoPBForwardDiff
        # Fail HERE, at construction, if the extension is not loaded. Without
        # this the object builds fine and then falls through to the generic
        # DifferentiationInterface path at gradient time, producing a
        # `MethodError: no method matching value_and_gradient(...)` naming a
        # dozen internal types — which says nothing about the actual problem.
        if !_piste_loaded()
            throw(ArgumentError(
                "AutoPBForwardDiff() needs the Piste package to be loaded. Add " *
                "`import Piste` (or `using Piste`) alongside `using PracticalBayes`.
" *
                "Piste is a weak dependency because `juliac --trim` — the reason " *
                "this backend exists — requires Julia 1.12+, so users on 1.10 are " *
                "not made to install it."))
        end
        # Our own forward-mode duals (forwarddiff.jl). No preparation of any
        # kind: there is no tape to trace and no config to build — the chunk
        # width is a type parameter and the seeds are constructed per call.
        # That absence IS the feature, since a runtime-built config is exactly
        # what stops ForwardDiff from surviving `juliac --trim`.
        return LogDensityFunction(model, layout, store, adtype, nothing, reject_errors)
    end
    # `DI.prepare_gradient` traces/compiles the AD backend's tape or config
    # ONCE against `θ0`'s type and length; every later `logdensity_and_gradient`
    # call reuses `prep` instead of re-preparing, which matters a lot for
    # tape-based reverse-mode backends (ReverseDiff) where retracing per call
    # would be far more expensive than the gradient itself.
    #
    # `model`, `layout`, and `store` are wrapped in `DI.Constant(...)`: this
    # tells the AD backend "these arguments are non-differentiable, treat them
    # as compile-time constants" — critically, this is what keeps latent
    # values in `store` invisible to the gradient regardless of which AD
    # backend is used (ForwardDiff never even sees them promoted to Dual;
    # reverse-mode backends never build a tape node for them).
    #
    # The layout goes in as its `EvalLayout` (see below), never the full
    # `Layout`: every AD constant is walked by reverse-mode backends on every
    # call, and the full Layout drags its site records along with it.
    prep = DI.prepare_gradient(_logdensity_call, adtype, θ0, DI.Constant(model), DI.Constant(EvalLayout(layout)), DI.Constant(store))
    return LogDensityFunction(model, layout, store, adtype, prep, reject_errors)
end

# Why not the full Layout: a reverse-mode backend builds and zeroes a tangent
# for every constant argument on every gradient call, and a `Layout` carries
# `meta::Vector{SiteRecord}` and a `Set` along with its slots. Each SiteRecord
# keeps its site's distribution exemplar and initial value, so for an observe
# site like `y ~ MvNormal(X * beta, ...)` that is an N-length vector. Measured
# under Mooncake on the normal-regression benchmark (Julia 1.12.7, Mooncake
# 0.5.67): the full Layout cost 8 extra allocations per gradient and a time
# penalty that grew with N (Turing 287 us, PB 345 us at N=20_000; a CPU profile
# put 12.7% of PB's gradient time in Mooncake zeroing those tangents). Passing
# only the slots gave exactly Turing's 17 allocations and its time (277 us),
# with identical values and gradients.
"""
    EvalLayout(layout::Layout)

The part of a [`Layout`](@ref) that model evaluation reads: its `slots`. Every
AD call receives this as its constant layout argument, not the `Layout`.
"""
struct EvalLayout{S<:NamedTuple}
    slots::S
end
EvalLayout(layout::Layout) = EvalLayout(layout.slots)

# Does this model qualify for the closed-form gradient? Returns a 1-tuple
# holding the prep, or an empty tuple — a tuple rather than `Union{Nothing,...}`
# so the result stays inferable.
#
# The three conditions are exactly `GradMode()`'s own: a plan must have been
# recognized at macro-expansion time, `store` must be empty (the closed form is
# derived for the whole model at once, so a Gibbs block cannot use it), and the
# response argument must be present.
function _maybe_gradmode(model, layout, store, θ0)
    isempty(store) || return ()
    plan = gradmode_plan(model.f)
    plan === nothing && return ()
    hasproperty(model.args, plan.response) || return ()
    N = length(getfield(model.args, plan.response))
    return (GradModePrep(plan, GradModeWorkspace{eltype(θ0)}(N), gradmode_cache(plan, layout)),)
end

# True once `import Piste` has activated `ext/PracticalBayesPisteExt.jl`.
# Asking the extension machinery directly is more honest than checking for the
# Piste module itself: what matters is whether the gradient METHOD exists.
_piste_loaded() = Base.get_extension(@__MODULE__, :PracticalBayesPisteExt) !== nothing

# The actual "run the model and read off the joint log-density" step, shared
# by both the density-only and density+gradient code paths below so there is
# exactly one place this logic lives (any bug fix here fixes both).
#
# `layout` is a `Layout` or its `EvalLayout`: evaluation reads only `.slots`.
# AD calls pass the `EvalLayout` (see its docstring); the density-only path
# passes the `Layout` it already holds.
@inline function _logdensity_call(θ, model::Model, layout, store::NamedTuple)
    mode = EvalMode(layout, θ, store, model.conditioned)
    _, acc = evaluate(model, mode, Accum(zero(eltype(θ))))
    # Coerce the result back to `eltype(θ)`. The accumulator promotes to the
    # widest type it sees (accumulator.jl), so a Float32 `θ` running a model
    # written with Float64 distribution literals — OR merely one whose *data*
    # (`y`/`X`) is Float64, which we explicitly document as allowed — produces
    # a Float64-primal logjoint even though the parameters are Float32. Plain
    # ForwardDiff/Mooncake/Enzyme tolerate that value/gradient type mismatch,
    # but PolyesterForwardDiff does not: its `threaded_gradient!` stores the
    # primal into a `Ref{eltype(θ)}` via `store_val!(::Ref{T}, ::T)`, which
    # `MethodError`s the moment the primal is Float64 and `θ` is Float32.
    # Coercing here makes the returned type track `eltype(θ)` exactly, fixing
    # PolyesterForwardDiff and costing nothing on the (already-matching) common
    # path. `_coerce_eltype` is a no-op when the types already agree.
    return _coerce_eltype(logjoint(acc), θ)
end

# `θ` under ForwardDiff-family backends is a `Vector{<:Dual}`, so `eltype(θ)`
# is that Dual type and `convert` correctly narrows a Float64-primal Dual to a
# Float32-primal one (primal AND partials). For a plain `Vector{Float32}`
# (density-only path) it's an ordinary `Float64 -> Float32` convert. The
# same-type case (`V === eltype(θ)`) hits the identity method and is elided.
@inline _coerce_eltype(v::V, θ::AbstractVector{V}) where {V} = v
@inline _coerce_eltype(v, θ) = convert(eltype(θ), v)

# Likelihood-only variant of `_logdensity_call`, used by `maximum_likelihood`
# (optimize.jl) — identical evaluation, just reads off `loglikelihood_(acc)`
# instead of `logjoint(acc)`. `Accum`'s prior/likelihood split (accumulator.jl)
# means this needs no new machinery in tilde.jl at all.
@inline function _loglikelihood_call(θ, model::Model, layout, store::NamedTuple)
    mode = EvalMode(layout, θ, store, model.conditioned)
    _, acc = evaluate(model, mode, Accum(zero(eltype(θ))))
    # Same `eltype(θ)` coercion as `_logdensity_call` above — keeps the
    # optimize.jl likelihood path usable under PolyesterForwardDiff too.
    return _coerce_eltype(loglikelihood_(acc), θ)
end

# Shared by both `reject_errors` branches below: an exception during
# evaluation (out-of-support `logpdf`, a failed `Cholesky`, etc.) is not a
# bug in the framework, it's a sign this particular `θ` is a bad draw — so we
# report it exactly the way an impossible point should be reported, `-Inf`
# density, rather than letting it crash the sampler.
@inline _reject_value(θ) = convert(eltype(θ), -Inf)

function LogDensityProblems.logdensity(f::LogDensityFunction, θ::AbstractVector)
    if f.reject_errors
        try
            return _logdensity_call(θ, f.model, f.layout, f.store)
        catch e
            e isa Union{InterruptException,OutOfMemoryError} && rethrow()
            return _reject_value(θ)
        end
    end
    return _logdensity_call(θ, f.model, f.layout, f.store)
end

LogDensityProblems.dimension(f::LogDensityFunction) = f.layout.dim

# LogDensityProblems uses `capabilities` (dispatched on the TYPE, not the
# value, hence `::Type{<:LogDensityFunction{...}}`) to decide whether a
# sampler needs to wrap this object in its own AD before it can get
# gradients. Since we've already attached gradients ourselves whenever
# `adtype !== nothing`, we report order 1 in that case so AdvancedHMC uses
# our `logdensity_and_gradient` method directly instead of re-differentiating.
function LogDensityProblems.capabilities(::Type{<:LogDensityFunction{<:Any,<:Any,<:Any,AD}}) where {AD}
    return AD === Nothing ? LogDensityProblems.LogDensityOrder{0}() : LogDensityProblems.LogDensityOrder{1}()
end

# AutoPBForwardDiff's `logdensity_and_gradient` lives in
# `ext/PracticalBayesPisteExt.jl`, because Piste is a weak dependency (see
# forwarddiff.jl for why). What stays here is the objective it differentiates,
# which needs no Piste types at all.
#
# `_PBFDObjective` is a callable struct rather than an anonymous closure ON
# PURPOSE. `θ -> _logdensity_call(θ, model, layout, store)` works fine under the
# JIT, but a closure over locals is what produced `Core.Box`-flavoured verifier
# errors while getting this to trim — every captured field inferring as `Any`.
# A struct with concrete fields captures the same three values with nothing
# boxed and every field read statically resolvable.
struct _PBFDObjective{M<:Model,L<:Layout,S<:NamedTuple}
    model::M
    layout::L
    store::S
end

@inline function (o::_PBFDObjective)(θ)
    return _logdensity_call(θ, o.model, o.layout, o.store)
end

# GradMode: the closed-form path. Same contract as the DI path below (returns
# `(value, gradient)`), and the same `reject_errors` semantics, so a sampler
# cannot tell the difference apart from speed.
function LogDensityProblems.logdensity_and_gradient(
    f::LogDensityFunction{M,L,S,GradMode,P}, θ::AbstractVector
) where {M<:Model,L<:Layout,S<:NamedTuple,P}
    g = similar(θ)
    if f.reject_errors
        try
            v = gradmode_value_and_gradient!(f.prep.plan, f.layout, f.model.args, θ, g, f.prep.ws, f.prep.cache)
            return v, g
        catch e
            e isa Union{InterruptException,OutOfMemoryError} && rethrow()
            return _reject_value(θ), zeros(eltype(θ), length(θ))
        end
    end
    v = gradmode_value_and_gradient!(f.prep.plan, f.layout, f.model.args, θ, g, f.prep.ws, f.prep.cache)
    return v, g
end

function LogDensityProblems.logdensity_and_gradient(f::LogDensityFunction, θ::AbstractVector)
    # `DI.value_and_gradient` reuses `f.prep` (built once at construction, see
    # above) and returns `(value, gradient)` — the exact pair
    # `LogDensityProblems.logdensity_and_gradient` is expected to return, so
    # there's nothing else to do here beyond passing the constants through
    # again (they must match what `prep` was built with).
    if f.reject_errors
        try
            return DI.value_and_gradient(
                _logdensity_call, f.prep, f.adtype, θ, DI.Constant(f.model), DI.Constant(EvalLayout(f.layout)), DI.Constant(f.store)
            )
        catch e
            e isa Union{InterruptException,OutOfMemoryError} && rethrow()
            # A zero gradient at `-Inf` is a safe "go nowhere" signal for any
            # sampler that checks the density before trusting the gradient
            # (e.g. NUTS's divergence check) — it never needs to be a real
            # ascent direction since the point is rejected outright anyway.
            return _reject_value(θ), zeros(eltype(θ), length(θ))
        end
    end
    return DI.value_and_gradient(
        _logdensity_call, f.prep, f.adtype, θ, DI.Constant(f.model), DI.Constant(EvalLayout(f.layout)), DI.Constant(f.store)
    )
end
