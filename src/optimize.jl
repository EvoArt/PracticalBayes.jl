using ADTypes: ADTypes, AbstractADType
using LinearAlgebra: LinearAlgebra, Symmetric
using Distributions: Distributions, MvNormal
import DifferentiationInterface as DI

# ===========================================================================
# Point estimation (MAP / MLE) and the Laplace approximation, built directly
# on top of `LogDensityFunction`'s two ingredients — a scalar objective and
# its gradient over the flat unconstrained `θ` — so none of this needs any
# M2/M3 sampling machinery. MAP maximizes the joint log-density
# (`_logdensity_call`, logdensity.jl); MLE maximizes the likelihood term only
# (`_loglikelihood_call`); Laplace is the MAP point plus the Hessian of the
# joint log-density there (`DI.hessian`, same pattern as the existing
# gradient plumbing).
#
# This is a secondary feature — full Bayesian inference (NUTS via
# AbstractMCMC/AdvancedHMC) is the main target of this package, so the
# optimizer itself is NOT reimplemented here: these functions require
# `using Optimization` (plus a solver package, e.g. `OptimizationOptimJL`)
# and hand off to `ext/PracticalBayesOptimizationExt.jl`. There is
# deliberately no built-in fallback optimizer.
# ===========================================================================

"""
    PointEstimate

The result of [`maximum_a_posteriori`](@ref) or [`maximum_likelihood`](@ref).

Fields: `constrained` (the estimate by name, on the parameters' own scale),
`theta` (the same point as the unconstrained vector), `value` (the log joint
or log-likelihood there), `retcode` (the optimiser's return code), and the
`layout` and `store` it was computed with.
"""
struct PointEstimate{T<:Real,V<:AbstractVector{T},L<:Layout,S<:NamedTuple,C<:NamedTuple}
    theta::V
    value::T
    constrained::C
    layout::L
    store::S
    retcode::Symbol
end

"""
    LaplaceApproximation

A normal approximation to the posterior on the unconstrained scale, from
[`laplace_approximation`](@ref). Fields: `map` (the [`PointEstimate`](@ref) it
is centred on), `hessian` of the log joint there, and `covariance`, which is
`-inv(hessian)`.
"""
struct LaplaceApproximation{T<:Real,M<:AbstractMatrix{T},PE<:PointEstimate{T}}
    covariance::M
    hessian::M
    map::PE
end

"""
    laplace_mvnormal(la::LaplaceApproximation) -> MvNormal

`la` as an `MvNormal` over the unconstrained vector `θ`. Throws if the
covariance is not positive definite, which usually means the optimiser did not
reach a maximum. Map draws back to named values with
`invlink(la.map.layout, θ)`.
"""
laplace_mvnormal(la::LaplaceApproximation) = MvNormal(la.map.theta, Symmetric(la.covariance))

# Builds the layout/store shared by all three point-estimation entry points.
# `store` here is USER-supplied fixed values (e.g. nuisance latents held
# constant during optimization); its keys are exactly the names routed to
# the value-store instead of the flat vector, mirroring how `values=` works
# in `build_layout` directly. `untracked` is forwarded as-is (see layout.jl).
function _build_point_layout(model, store, untracked, rng, init)
    layout, θ0, store0 = build_layout(model; values=Tuple(keys(store)), untracked=untracked, rng=rng, init=init)
    return layout, θ0, merge(store0, store)  # user-supplied values win over trace defaults
end

"""
    _external_optimize(negf, negg!, θ0, alg; kwargs...) -> (θ_opt, minimized_value, retcode)

Implemented by `ext/PracticalBayesOptimizationExt.jl`, loaded automatically
once the caller has `using Optimization` (a `[weakdeps]` entry — see
Project.toml). `negf(θ)` is the NEGATED objective (Optimization.jl minimizes;
we maximize) and `negg!(g, θ)` mutates `g` in place with the negated
gradient at `θ`.

Implemented as a `Ref`-held function (`_EXTERNAL_OPTIMIZE_HOOK`, below)
rather than an ordinary generic function the extension adds a method to: a
method with this exact signature defined in both this package and the
extension would be the SAME method to Julia, not a new dispatch, and
overwriting a method during another module's precompilation is forbidden
("Method overwriting is not permitted during Module precompilation").
Assigning into a `Ref` in the extension's `__init__` sidesteps this
entirely — it's a plain value replacement, not a method redefinition.
"""
function _external_optimize(negf, negg!, θ0, alg; kwargs...)
    return _EXTERNAL_OPTIMIZE_HOOK[](negf, negg!, θ0, alg; kwargs...)
end

# Default hook: errors with instructions. Replaced by
# `ext/PracticalBayesOptimizationExt.jl`'s `__init__` when `using Optimization`
# is loaded — there is no other implementation, so point estimation always
# requires it.
const _EXTERNAL_OPTIMIZE_HOOK = Ref{Any}(
    (negf, negg!, θ0, alg; kwargs...) -> error(
        "PracticalBayes' point-estimation functions (maximum_a_posteriori/" *
        "maximum_likelihood/laplace_approximation) require Optimization.jl. Run " *
        "`using Optimization` plus a solver package (e.g. `using OptimizationOptimJL`) " *
        "and pass an algorithm via `optimizer=...` (e.g. `OptimizationOptimJL.BFGS()`).",
    ),
)

# Shared driver for MAP and MLE: prepares one DI gradient tape/config against
# `objective_call` (either `_logdensity_call` or `_loglikelihood_call`, both
# logdensity.jl) and hands off to `_external_optimize` (Optimization.jl
# extension).
function _optimize_point(objective_call, model, layout, store, θ0::AbstractVector, adtype, optimizer; kwargs...)
    # The slots-only `EvalLayout` as the AD constant, as in logdensity.jl.
    elayout = EvalLayout(layout)
    prep = DI.prepare_gradient(objective_call, adtype, θ0, DI.Constant(model), DI.Constant(elayout), DI.Constant(store))
    fg(θ) = DI.value_and_gradient(objective_call, prep, adtype, θ, DI.Constant(model), DI.Constant(elayout), DI.Constant(store))

    # Optimization.jl (like most external optimizers) minimizes; we maximize,
    # so negate. `negg!` recomputes the value via `fg` too (and discards it)
    # rather than needing a separate gradient-only AD call — this runs once
    # per optimizer iteration for a one-off point estimate, not inside any
    # hot loop, so the extra scalar is not worth a second code path.
    negf(θ) = -fg(θ)[1]
    function negg!(g, θ)
        _, grad = fg(θ)
        g .= .-grad
        return g
    end
    θ_hat, negval, retcode = _external_optimize(negf, negg!, θ0, optimizer; kwargs...)
    return θ_hat, -negval, retcode
end

"""
    maximum_a_posteriori(model, optimizer; init=NamedTuple(), store=NamedTuple(),
                         adtype=AutoForwardDiff(), kwargs...) -> PointEstimate

Find the posterior mode of `model`, optimising over the unconstrained
parameters. Needs `using Optimization` and a solver package; `optimizer` is a
solver such as `OptimizationOptimJL.LBFGS()`.

# Keywords
- `init`: starting values by name.
- `store`: values, by name, of parameters to hold fixed.
- `adtype`: AD backend for the gradient.
- Other keywords go to `Optimization.solve`.

```julia
using Optimization, OptimizationOptimJL
est = maximum_a_posteriori(model, LBFGS())
est.constrained
```
"""
function maximum_a_posteriori(
    model,
    optimizer;
    store=NamedTuple(),
    untracked=(),
    rng=Random.default_rng(),
    init=NamedTuple(),
    adtype=ADTypes.AutoForwardDiff(),
    kwargs...,
)
    layout, θ0, full_store = _build_point_layout(model, store, untracked, rng, init)
    θ_hat, val, retcode = _optimize_point(_logdensity_call, model, layout, full_store, θ0, adtype, optimizer; kwargs...)
    constrained = invlink(layout, θ_hat; include_untracked=true)
    return PointEstimate(θ_hat, val, constrained, layout, full_store, retcode)
end

"""
    maximum_likelihood(model, optimizer; kwargs...) -> PointEstimate

Like [`maximum_a_posteriori`](@ref), but maximise the log-likelihood alone,
ignoring the priors.
"""
function maximum_likelihood(
    model,
    optimizer;
    store=NamedTuple(),
    untracked=(),
    rng=Random.default_rng(),
    init=NamedTuple(),
    adtype=ADTypes.AutoForwardDiff(),
    kwargs...,
)
    layout, θ0, full_store = _build_point_layout(model, store, untracked, rng, init)
    θ_hat, val, retcode = _optimize_point(_loglikelihood_call, model, layout, full_store, θ0, adtype, optimizer; kwargs...)
    constrained = invlink(layout, θ_hat; include_untracked=true)
    return PointEstimate(θ_hat, val, constrained, layout, full_store, retcode)
end

"""
    laplace_approximation(model, optimizer; kwargs...) -> LaplaceApproximation

Find the posterior mode with [`maximum_a_posteriori`](@ref) (same arguments)
and fit a normal approximation there, with covariance minus the inverse
Hessian of the log joint. The approximation is on the unconstrained scale.
"""
function laplace_approximation(
    model,
    optimizer;
    store=NamedTuple(),
    untracked=(),
    rng=Random.default_rng(),
    init=NamedTuple(),
    adtype=ADTypes.AutoForwardDiff(),
    kwargs...,
)
    map_est = maximum_a_posteriori(
        model, optimizer; store=store, untracked=untracked, rng=rng, init=init, adtype=adtype, kwargs...
    )
    H = DI.hessian(
        _logdensity_call, adtype, map_est.theta, DI.Constant(model), DI.Constant(EvalLayout(map_est.layout)), DI.Constant(map_est.store)
    )
    Σ = Matrix(Symmetric(-inv(H)))
    return LaplaceApproximation(Σ, Matrix(H), map_est)
end
