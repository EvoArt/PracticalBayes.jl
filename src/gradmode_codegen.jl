# =============================================================================
# Stage 2, part 2: closed-form value+gradient evaluation from a `GLMPlan`.
#
# `gradmode_value_and_gradient!(plan, layout, args, theta, g, ws)` computes the
# joint log-density and its gradient analytically, for a model that
# `recognize_glm` accepted. Stage 1 measured this shape of gradient at
# 5.45x-8.55x faster than Mooncake with the real Layout indirection included.
#
# STRUCTURE OF THE COMPUTATION
# ----------------------------
# Every recognized model has the same three-part shape, which is exactly why
# a closed form exists:
#
#   1. read parameters out of theta (through the real Layout slots/bijectors)
#   2. eta = sum of linear terms;  loglik = sum_i f(eta_i, y_i)
#   3. gradient: dL/d(param) = (dEta/d(param))' * (dL/dEta), plus prior terms
#
# The only per-link knowledge needed is `dL/dEta`, which for a GLM is a
# one-liner (see `_gm_link_loglik_and_dEta!`). The chain back to each
# parameter is then determined entirely by the term kind, NOT by the link:
#   :intercept  -> sum(w)
#   :matvec     -> X' * w
#   :scaled     -> dot(x, w)
#   :indexed    -> scatter-add w into groups
# That factorisation is what makes this general across links rather than one
# hardcoded model.
#
# WHY THIS IS AN INTERPRETER, NOT A MACRO
# ---------------------------------------
# It walks `plan.terms` at run time rather than generating specialized code
# per model. The loops are over a handful of terms whose bodies are the O(N)
# BLAS/SIMD work, so the dispatch overhead is negligible against the matvec —
# Stage 1 confirmed the achievable time is dominated by `mul!`. Generating
# specialized code is a possible later optimization, not a prerequisite, and
# keeping it an interpreter makes it far easier to verify.
#
# SAFETY: nothing here is trusted on its own. `check_gradmode` (below) compares
# this gradient against a general-AD gradient at randomized points, mirroring
# `check_depends.jl`'s role for `depends=` annotations. Recognition plus
# codegen is an optimization; the check is what licenses using it.
#
# STATUS: wired in. `LogDensityFunction` checks `gradmode_plan` and uses the
# closed form automatically for a recognised model (3fec222), so passing
# `GradMode()` explicitly is only needed to make the requirement strict --
# an explicit request throws when the model is not recognised, where the
# automatic path just leaves the requested backend in place.
# =============================================================================

# FUTURE WORK: GradMode as a per-site RULE rather than a whole-model swap.
#
# Today this is all-or-nothing at model level: `LogDensityFunction` checks
# `gradmode_plan` and, if the model is recognised, swaps the requested AD
# backend for `GradMode()` entirely. Unrecognised models get no benefit at all,
# even if they contain a GLM-shaped part.
#
# The alternative, not built: make the closed form a RULE that reverse-mode AD
# consults per site, so a recognised block inside a larger model gets the
# analytic gradient while everything around it is taped normally. That is what
# Stan does with `precomputed_gradients`, and it would compose where this does
# not.
#
# Worth testing once the benchmark work is finished: does it still pay when the
# full model is more complex than the recognised part? The whole-model swap is
# ~9x faster than the fastest general AD on a pure GLM, but that says nothing
# about a model where the GLM is one block among several -- the tape's
# bookkeeping for everything else might swamp the saving. Measure before
# building.

"""
    GradMode()

Backend that computes the gradient in closed form, for models recognised as
generalised linear models. [`LogDensityFunction`](@ref) already uses it for a
recognised model whatever backend you ask for; pass `GradMode()` to make that a
requirement, so that construction throws if the model is not recognised.

A model is recognised when its linear predictor is visibly linear in the
parameters: a sum of `X * beta` (data matrix times parameter vector),
`x .* b` (data vector times scalar parameter), `a[idx]` (a parameter indexed
by data), a scalar parameter, or data. A product of two parameters is not
recognised. It cannot be used for a [`Gibbs`](@ref) block that holds only part
of the model.

```julia
@model function poisson_reg(X, y)
    beta ~ MvNormal(zeros(size(X, 2)), I)
    y ~ arraydist(LogPoisson.(X * beta))
end
```

See also [`is_gradmode_compatible`](@ref), [`check_gradmode`](@ref).
"""
struct GradMode end

"""
    gradmode_plan(f) -> Union{GLMPlan,Nothing}

The `GLMPlan` recognized for a model evaluator at `@model` expansion time, or
`nothing` when the model was not recognized (or `f` is not a model evaluator).
"""
gradmode_plan(f) = _model_info(f).plan

"""
    is_gradmode_compatible(model::Model) -> Bool

Whether `model` was recognised as a generalised linear model with a
closed-form gradient (see [`GradMode`](@ref)). `false` means the model will use
ordinary AD, not that it is not a GLM.

```jldoctest
julia> @model function reg(x, y)
           b ~ Normal(0, 1)
           y ~ MvNormal(x .* b, 1.0)
       end;

julia> is_gradmode_compatible(reg([1.0, 2.0], [0.5, 1.5]))
true
```
"""
is_gradmode_compatible(model::Model) = gradmode_plan(model.f) !== nothing

"""
    GradModeWorkspace(N, dim)

Pre-allocated scratch for `gradmode_value_and_gradient!`: `eta` and the
per-observation weight vector `w = dL/dEta`, both length `N`. Held across
calls so the hot path allocates nothing (an HMC leapfrog calls this on every
step, so per-call allocation would show up immediately).
"""
struct GradModeWorkspace{T}
    eta::Vector{T}
    w::Vector{T}
    # Site name => constrained value, refilled on every call. Kept here rather
    # than built per call: a fresh Dict cost an allocation and a rehash per
    # gradient, which is a visible share of a small model's ~1 us budget.
    vals::Dict{Symbol,Any}
end
GradModeWorkspace{T}(N::Int) where {T} =
    GradModeWorkspace{T}(zeros(T, N), zeros(T, N), Dict{Symbol,Any}())

# --- per-model cache --------------------------------------------------------
#
# Everything about a prior site that does not depend on theta, worked out ONCE
# when the LogDensityFunction is built instead of on every gradient.
#
# Measured on a 50x2 normal regression (Athena, 2026-11-04): a GradMode
# gradient took 3.9 us against 0.9 us for ForwardDiff and 0.29 us for the plain
# log density. The arithmetic was ~0.15 us of that. The rest was per-call
# lookup that never changes between calls: a scan of `layout.meta` for each
# site's distribution (twice per site), the generic Bijectors call on that
# untyped distribution (~1.4 us for two sites, dynamically dispatched), a
# `try minimum/maximum` support check, and a walk of the prior's AST for its
# numeric constants.
const GM_GENERIC    = Int8(0)   # anything else: the generic Bijectors path
const GM_ID_SCALAR  = Int8(1)   # scalar, support (-Inf, Inf): x = z
const GM_EXP_SCALAR = Int8(2)   # scalar, support [0, Inf):    x = exp(z), logjac = z
const GM_ID_VECTOR  = Int8(3)   # MvNormal-family vector:      x = z

struct GMPrior
    range::UnitRange{Int}
    # The Layout's recorded distribution and its linked->constrained bijector,
    # kept for the GM_GENERIC path (and for `kind` to be checked against).
    dist::Any
    bij::Any
    kind::Int8
    positive::Bool
    # The prior's numeric constants, read off the AST once:
    #   Normal (mu, sd)   Cauchy (mu, scale)   Exponential (theta, -)
    #   MvNormal (c, -) for c*I   Flat/FlatPos and hierarchical sites: unused
    c1::Float64
    c2::Float64
end

struct GradModeCache
    priors::Vector{GMPrior}     # parallel to `plan.priors`
    scale_index::Int            # theta index of the observation scale; 0 if none
end

"""
    gradmode_cache(plan, layout) -> GradModeCache

The theta-independent part of every prior site in `plan`, for
`gradmode_value_and_gradient!`.

Each site's transform is classified from the Layout's own distribution
(identity, `exp`, or neither) and the classification is CHECKED: the fast
transform is evaluated against the generic Bijectors one at a test point, and a
site where the two disagree falls back to the generic path. A mismatch there
would be a silently wrong posterior, so it is verified rather than assumed.
"""
function gradmode_cache(plan::GLMPlan, layout)
    priors = GMPrior[]
    for ps in plan.priors
        slot = getproperty(layout.slots, ps.name)
        slot isa FlatSlot || error("gradmode: expected FlatSlot for $(ps.name)")
        r = slot.range
        d = _gm_exemplar(ps, layout)
        bij = from_linked_vec(d)
        kind = _gm_check_kind(_gm_classify(d, r), bij, r)
        c1, c2 = _gm_is_hier(ps) ? (NaN, NaN) : _gm_prior_constants(ps)
        push!(priors, GMPrior(r, d, bij, kind, _gm_positive_support(d), c1, c2))
    end
    scale_index = 0
    if plan.link in (:Normal, :MvNormal) && plan.scale !== nothing
        scale_index = first(getproperty(layout.slots, plan.scale).range)
    end
    return GradModeCache(priors, scale_index)
end

function _gm_classify(d, r)
    if d isa Distributions.UnivariateDistribution && length(r) == 1
        lo, hi = try
            (minimum(d), maximum(d))
        catch
            return GM_GENERIC
        end
        (lo == -Inf && hi == Inf) && return GM_ID_SCALAR
        (lo == 0 && hi == Inf) && return GM_EXP_SCALAR
    elseif d isa Distributions.AbstractMvNormal && length(d) == length(r)
        return GM_ID_VECTOR
    end
    return GM_GENERIC
end

# Evaluate the fast transform against the generic one at an arbitrary point;
# keep the fast one only if they agree to rounding.
function _gm_check_kind(kind, bij, r)
    kind === GM_GENERIC && return kind
    theta = [0.37 * sin(1.3i) - 0.21 for i in 1:last(r)]
    xg, lg = with_logabsdet_jacobian(bij, _linked_view(theta, r))
    xf, lf = _gm_fast_read(kind, r, theta)
    ok = size(xf) == size(xg) &&
         isapprox(vec(collect(xf)), vec(collect(xg)); rtol=1e-12, atol=1e-14) &&
         isapprox(lf, lg; rtol=1e-12, atol=1e-14)
    return ok ? kind : GM_GENERIC
end

@inline function _gm_fast_read(kind, r, theta)
    if kind === GM_ID_SCALAR
        x = theta[first(r)]
        return x, zero(x)
    elseif kind === GM_EXP_SCALAR
        z = theta[first(r)]
        return exp(z), z
    else # GM_ID_VECTOR
        return _linked_view(theta, r), zero(eltype(theta))
    end
end

function _gm_prior_constants(ps::PriorSite)
    d = ps.dist
    d === :Normal && return (_gm_num(ps.args, 1, 0.0), _gm_num(ps.args, 2, 1.0))
    d === :Cauchy && return (_gm_num(ps.args, 1, 0.0), _gm_num(ps.args, 2, 1.0))
    d === :Exponential && return (_gm_num(ps.args, 1, 1.0), NaN)
    d === :MvNormal && return (_gm_mvnormal_var_of(ps), NaN)
    return (NaN, NaN)   # Flat / FlatPos: no constants
end

"""
    GradModePrep(plan, ws, cache)

What `LogDensityFunction` stores in its `prep` slot for a `GradMode()` backend,
in place of a `DifferentiationInterface` tape/config. Holds the recognized plan,
the reusable scratch buffers and the per-model [`gradmode_cache`](@ref), so the
hot path allocates only the returned gradient vector.
"""
struct GradModePrep{P<:GLMPlan,W}
    plan::P
    ws::W
    cache::GradModeCache
end

"""
    gradmode_value_and_gradient!(plan, layout, args, theta, g, ws[, cache]) -> value

Closed-form log-joint and gradient for a recognized GLM. Writes the gradient
into `g` (which must be `length(theta)`) and returns the log-joint.

`args` is the model's argument NamedTuple (`Model.args`), from which data
operands are looked up by the names the recognizer recorded. `cache` is
[`gradmode_cache`](@ref)`(plan, layout)`; pass it when calling repeatedly, since
building it is most of what a small model's gradient used to cost.
"""
function gradmode_value_and_gradient!(plan::GLMPlan, layout, args, theta, g, ws,
                                      cache::GradModeCache=gradmode_cache(plan, layout))
    T = eltype(theta)
    fill!(g, zero(T))

    # --- 1. parameters, read through the real Layout ------------------------
    # `_gm_read_param` goes through the same bijector call `_assume` uses, so
    # constrained parameters (e.g. Exponential -> exp) are handled identically
    # and the log-Jacobian is accumulated the same way.
    # `vals` maps a site name to its CONSTRAINED value. `Any` because the
    # values are heterogeneous: a scalar site gives a `Float64`, a vector site
    # gives a `SubArray` (NOT a `Vector` — `_linked_view` deliberately
    # constructs a `Base.SubArray` to dodge AD backends that overload `view`;
    # see its docstring in tilde.jl). The element type is not fixed either:
    # `theta` may be Float32 (this package is Float32-first) or a Dual under a
    # ForwardDiff-family backend.
    #
    # KNOWN COST, not an oversight: this `Any` is why `_gm_hier_params` infers
    # `Tuple{Any,Any}`, which blocks `juliac --trim=safe` (162 verifier errors
    # cascade from here) and costs some dynamic dispatch on the prior/scale
    # arithmetic. The fix is NOT to annotate at the use site — hyper-locations
    # can legitimately be vectors, and asserting `::Float64` there is what
    # would have reintroduced the wrong gradient fixed in e31e113. The right
    # fix is to stop using a Dict: the name set is fixed at recognition time
    # (`plan.priors`), so these could live in a NamedTuple or a
    # position-indexed Vector, concrete by construction rather than by
    # assertion. Would also need a recognizer-side rejection for any site whose
    # linked representation is not scalar-or-vector, so the invariant is
    # ENFORCED rather than assumed — skipping that half is how the
    # vector-hyper-location bug (e31e113) happened in the first place.
    #
    # PARKED, deliberately (decision 2026-09-04). The only thing it unlocks is
    # `--trim=safe` for GradMode-recognized models, and trimming already works
    # today for anyone willing to hand-write a gradient (204 -> 0 errors, 2.9 MB
    # binary matching the JIT to 16 digits). So this
    # is convenience, not capability. The dispatch cost it also removes is
    # small: profiling puts the closed form at 79.6% BLAS `gemv!` at
    # N=20000/P=200, so the prior arithmetic is not where the time goes.
    #
    # Do NOT gate this on `@static_model` if it is ever picked up. That was
    # considered and rejected: `vals` threads through ten functions, so a
    # conditional version means either duplicating the whole gradient path (two
    # copies that can drift, in the file where a mistake silently biases a
    # posterior) or parameterising the container — which IS the general fix, at
    # which point gating buys nothing. `@static_model` is opt-in because it
    # costs ~5ms compile time per site record; this has no such tradeoff and is
    # strictly better for every model, so it should be unconditional or not at
    # all.
    vals = ws.vals
    empty!(vals)
    lp = zero(T)
    for (k, ps) in enumerate(plan.priors)
        pc = cache.priors[k]
        v, ljac = _gm_read_param(pc, theta)
        vals[ps.name] = v
        lp += (_gm_is_hier(ps) ? _gm_hier_logpdf(ps, v, vals) : _gm_prior_logpdf(ps, pc, v)) + ljac
    end

    # --- 2. forward: eta ----------------------------------------------------
    eta = ws.eta
    fill!(eta, zero(T))
    for t in plan.terms
        _gm_accum_term!(eta, t, vals, args)
    end

    # --- 3. likelihood value and dL/dEta ------------------------------------
    y = getfield(args, plan.response)
    sigma = _gm_obs_scale(plan, vals)
    ll = _gm_link_loglik_and_dEta!(ws.w, plan.link, eta, y, sigma)
    lp += ll

    # --- 4. reverse: push w back through each term to its parameter ---------
    for t in plan.terms
        _gm_pullback_term!(g, t, ws.w, vals, args, layout)
    end

    # --- 5. prior gradient contributions ------------------------------------
    for (k, ps) in enumerate(plan.priors)
        if _gm_is_hier(ps)
            _gm_hier_grad!(g, ps, vals[ps.name], vals, layout)
        else
            _gm_prior_grad!(g, ps, cache.priors[k], vals[ps.name])
        end
    end

    # the observation scale (sigma) gets its own likelihood-side derivative
    if cache.scale_index != 0
        _gm_scale_grad!(g, cache.scale_index, vals[plan.scale], eta, y)
    end
    return lp
end

# --- parameter reading ------------------------------------------------------

# Reads one prior's parameter out of theta via its Layout slot, using the same
# bijector path as `_assume(::FlatSlot, ...)` in tilde.jl so constrained
# parameters and their log-Jacobians behave identically.
#
# The two common transforms (identity, and `exp` onto [0, Inf)) are done
# directly; `gradmode_cache` has already checked them against Bijectors for
# this site. They also keep theta's element type by construction, so the width
# fix below is only needed on the generic path.
function _gm_read_param(pc::GMPrior, theta)
    pc.kind === GM_GENERIC || return _gm_fast_read(pc.kind, pc.range, theta)
    x, ljac = with_logabsdet_jacobian(pc.bij, _linked_view(theta, pc.range))
    # The exemplar is recorded ONCE by TraceMode and carries the element type of
    # the distribution as written (`Exponential(1)` is Float64), so Bijectors
    # returns a Float64 `x` even when `theta` is Float32 — and everything
    # downstream then promotes, running the whole closed form in Float64.
    #
    # The general-AD path does not have this problem because it re-evaluates the
    # model body at the current theta, so its distributions take theta's type.
    # Here the exemplar is deliberately reused (see `_gm_exemplar`), so the
    # width has to be restored explicitly. Note this is a WIDTH conversion only:
    # the bijector has already run, so no derivative information is discarded.
    #
    # Caught by test/ad_backends.jl's `@test vref isa Float32`, which started
    # failing when the closed form became automatic for recognised models.
    T = float(eltype(theta))
    return _gm_toT(T, x), _gm_toT(T, ljac)
end

# Convert a site value (scalar, or a vector/view of them) to width `T`, leaving
# anything that is not a plain float alone — a `Dual` under a ForwardDiff-family
# backend must pass through untouched, which is why this dispatches on
# `AbstractFloat` rather than converting unconditionally.
@inline _gm_toT(::Type{T}, x::AbstractFloat) where {T} = convert(T, x)
@inline _gm_toT(::Type{T}, x::AbstractArray{<:AbstractFloat}) where {T} =
    eltype(x) === T ? x : convert(AbstractArray{T}, x)
@inline _gm_toT(::Type{T}, x) where {T} = x

# The distribution instance for a site, taken from the Layout's own recorded
# exemplar (built during TraceMode) rather than re-evaluated from the AST —
# so the values used here are exactly the ones the normal path would use.
function _gm_exemplar(ps::PriorSite, layout)
    for rec in layout.meta
        rec.name === ps.name && return rec.dist_exemplar
    end
    error("gradmode: no layout record for $(ps.name)")
end

# --- prior value/gradient ---------------------------------------------------

# The working element type for one site's arithmetic.
#
# Everything the AST walk produces (`_gm_num`, `_gm_mvnormal_var_of`) is
# Float64 by construction — see `_gm_num`'s own comment for why that
# annotation is load-bearing. But this package is Float32-first, so a Float64
# CONSTANT multiplied into a Float32 parameter promotes the whole expression,
# and the closed form then silently runs the entire log-density in Float64.
# That is not just a slower path: `logdensity_and_gradient` returned a Float64
# value alongside a Float32 gradient, which is what caught this.
#
# So: keep the AST boundary at Float64, and convert to the parameter's own
# width here, at the point of use. `_gm_T` reads that width off the value being
# scored; `_gm_c` is the conversion.
@inline _gm_T(v) = float(eltype(v))
@inline _gm_c(::Type{T}, x) where {T} = convert(T, x)

# 2pi at the working width. Written this way rather than as `log(2pi)` because
# the literal is Float64 and would promote the sum it lands in.
@inline _gm_log2pi(::Type{T}) where {T} = log(_gm_c(T, 2) * _gm_c(T, pi))

_gm_prior_logpdf(ps::PriorSite, pc::GMPrior, v) =
    _gm_flatlike(ps.dist) ? zero(_gm_T(v)) : logpdf_of(ps, pc, v)

# Flat/FlatPos are constant on their support AND need no bounded transform, so
# they contribute nothing that varies with theta.
#
# `Uniform` USED to be in this list and it was a bug: it is bounded, so it is
# sampled through a logistic bijector whose log-Jacobian does vary with theta.
# Treating it as flat dropped that term from both the density and the gradient.
# It is now rejected by the recognizer instead (see GRADMODE_PRIORS).
_gm_flatlike(d::Symbol) = d in (:Flat, :FlatPos)

# The constants come from `pc` (`gradmode_cache`), read off the AST once rather
# than re-walked here on every call.
function logpdf_of(ps::PriorSite, pc::GMPrior, v)
    d = ps.dist
    # hierarchical (centered): location and/or scale come from `vals`, not the
    # AST, so this is handled by `_gm_hier_logpdf` with the resolved numbers.
    _gm_is_hier(ps) && error("gradmode: hierarchical prior needs _gm_hier_logpdf")
    T = _gm_T(v)
    half = _gm_c(T, 0.5)
    if d === :Normal
        mu, sd = _gm_c(T, pc.c1), _gm_c(T, pc.c2)
        return sum(@. -half*((v-mu)/sd)^2 - log(sd) - half*_gm_log2pi(T))
    elseif d === :MvNormal
        # isotropic covariance `c*I`; `c` recovered from the AST at
        # recognition time (see _gm_mvnormal_var). Ignoring it silently
        # differentiates a scaled prior as if it were standard normal.
        c = _gm_c(T, pc.c1)
        ss = zero(T)
        @inbounds for x in v; ss += x*x; end
        return -half*ss/c - half*_gm_c(T, length(v))*(_gm_log2pi(T) + log(c))
    elseif d === :Exponential
        th = _gm_c(T, pc.c1)
        return sum(@. -v/th - log(th))
    elseif d === :Cauchy
        mu, sc = _gm_c(T, pc.c1), _gm_c(T, pc.c2)
        return sum(@. -log(_gm_c(T, pi)*sc*(1 + ((v-mu)/sc)^2)))
    end
    error("gradmode: no closed-form logpdf for $(d)")
end

# d(logpdf)/dx at one element `x` of a non-hierarchical site, for each
# whitelisted prior. Flat-like priors contribute nothing from their density.
@inline function _gm_dlogp(d::Symbol, ::Type{T}, c1, c2, x) where {T}
    if d === :Normal
        mu, sd = _gm_c(T, c1), _gm_c(T, c2)
        return -(x - mu)/sd^2
    elseif d === :MvNormal
        return -x/_gm_c(T, c1)
    elseif d === :Exponential
        return -inv(_gm_c(T, c1))
    elseif d === :Cauchy
        mu, sc = _gm_c(T, c1), _gm_c(T, c2)
        return -2*(x - mu)/(sc^2 + (x - mu)^2)
    elseif _gm_flatlike(d)
        return zero(T)
    end
    error("gradmode: no closed-form prior gradient for $(d)")
end

# One element of a site value that may be a scalar or a vector.
@inline _gm_at(v::AbstractArray, k) = @inbounds v[k]
@inline _gm_at(v, k) = v

# d(logpdf)/dx for each whitelisted prior, written into g at the slot range,
# chained through the bijector: for an identity site `g += dx`; for a positive
# site under x = exp(z), `g += dx * x + 1`, the `+ 1` being the derivative of the
# log-Jacobian `z`.
#
# A flat-like prior has no density term, but a positive one still has the
# Jacobian's `+ 1`. Skipping it was a real bug: `sigma ~ FlatPos(0.0)` lost
# exactly 1.0 from its gradient while the density stayed right (PosteriorDB
# blr pattern) -- correct-looking output, wrong posterior.
#
# Element by element, with no temporary array: the broadcast form allocated a
# vector the size of the site on every call.
function _gm_prior_grad!(g, ps::PriorSite, pc::GMPrior, v)
    T = _gm_T(v)
    d = ps.dist
    if pc.positive
        @inbounds for (k, i) in enumerate(pc.range)
            x = _gm_at(v, k)
            g[i] += _gm_dlogp(d, T, pc.c1, pc.c2, x) * x + one(T)
        end
    else
        @inbounds for (k, i) in enumerate(pc.range)
            g[i] += _gm_dlogp(d, T, pc.c1, pc.c2, _gm_at(v, k))
        end
    end
    return nothing
end

# --- hierarchical (centered) priors ----------------------------------------
#
# `alpha ~ Normal(mu, sigma)` where mu and/or sigma are themselves parameters.
# All three gradients are needed, and omitting the hyperparameter ones is the
# classic silent-bias bug (the posterior for mu/sigma would be wrong while
# alpha's looked fine):
#
#   dL/dalpha = -(alpha - mu)/sigma^2
#   dL/dmu    =  sum((alpha - mu)/sigma^2)
#   dL/dsigma =  sum(((alpha-mu)^2/sigma^2 - 1)/sigma)
#
# Each hyperparameter's own bijector chain is then applied by
# `_gm_add_hyper!` (sigma is positive, so d/d(log sigma) = sigma * d/d sigma).

_gm_is_hier(ps::PriorSite) = ps.hyper_loc !== nothing || ps.hyper_scale !== nothing

# Resolved (mu, sigma) for a hierarchical site: from `vals` when the argument
# is a parameter, otherwise the literal from the AST.
# `mu` may be SCALAR (a hyper-mean broadcast over the site, `fill(mu_a, J)`)
# or a VECTOR (`a ~ MvNormal(mu, I)` with `mu` itself a vector parameter).
# `_gm_scalar` collapses the vector case to `mu[1]`, which silently uses one
# element for every observation — a wrong gradient, not a slow one. So the
# vector case is kept as a vector and the arithmetic below broadcasts.
#
# Found while investigating an unrelated type-inference question: annotating
# this `::Float64` (the obvious fix for the inference problem) would have been
# WRONG precisely because of this case, and checking whether the annotation
# was safe is what surfaced the bug.
#
# `T` is the working element type (see `_gm_T`). It applies ONLY to the
# constants recovered from the AST — a hyper value read out of `vals` already
# carries the parameter's own type and is passed through untouched, vector case
# included.
function _gm_hier_params(ps::PriorSite, vals, T)
    mu = ps.hyper_loc === nothing ?
        _gm_c(T, ps.dist === :MvNormal ? 0.0 : _gm_num(ps.args, 1, 0.0)) :
        _gm_hyper_value(vals[ps.hyper_loc])
    if ps.hyper_scale === nothing
        sd = _gm_c(T, ps.dist === :MvNormal ?
                      sqrt(_gm_mvnormal_var_of(ps)) : _gm_num(ps.args, 2, 1.0))
    else
        s = _gm_scalar(vals[ps.hyper_scale])
        # What the argument MEANS differs by distribution, and getting it
        # backwards is a silent factor error in the gradient:
        #   Normal(mu, s)        -> s is a standard deviation
        #   MvNormal(mu, s^2*I)  -> s^2 is a variance, so sigma = s
        #   MvNormal(mu, s*I)    -> s is the variance, so sigma = sqrt(s)
        # The recognizer recorded whether the expression was squared
        # (`_gm_scale_is_variance`) precisely so this does not have to guess.
        sq = _gm_scale_is_variance(get(ps.args, 2, nothing))
        sd = ps.dist === :Normal ? s : (sq ? s : sqrt(s))
    end
    return mu, sd
end

function _gm_hier_logpdf(ps::PriorSite, v, vals)
    T = _gm_T(v)
    mu, sd = _gm_hier_params(ps, vals, T)
    n = _gm_c(T, length(v))
    half = _gm_c(T, 0.5)
    #  is scalar or elementwise;  handles both without
    # allocating a broadcast temporary.
    ss = zero(T)
    @inbounds for k in eachindex(v); ss += (v[k] - _gm_mu_at(mu, k))^2; end
    return -half*ss/sd^2 - n*(log(sd) + half*_gm_log2pi(T))
end

# One element of a hyper-location that may be a scalar or a vector.
@inline _gm_mu_at(mu::Real, k) = mu
@inline _gm_mu_at(mu::AbstractArray, k) = @inbounds mu[k]

# A hyperparameter value: scalars stay scalar, length-1 vectors collapse (a
# scalar site is stored as a 1-element view), longer vectors stay vectors.
@inline function _gm_hyper_value(v)
    v isa AbstractArray || return v
    return length(v) == 1 ? @inbounds(v[1]) : v
end

function _gm_hier_grad!(g, ps::PriorSite, v, vals, layout)
    T = _gm_T(v)
    mu, sd = _gm_hier_params(ps, vals, T)
    s2 = sd*sd
    # elementwise in both cases: broadcasting a scalar mu is a no-op, and a
    # vector mu lines up with v element for element
    dv = [-(v[k] - _gm_mu_at(mu, k))/s2 for k in eachindex(v)]
    slot = getproperty(layout.slots, ps.name)
    _gm_add_chained!(g, slot.range, dv, v, ps, layout)

    # hyper-location. For a SCALAR hyper-mean every element contributes to the
    # same partial, so the contributions sum:  dL/dmu = sum((x - mu)/sigma^2).
    # For a VECTOR hyper-mean each element has its OWN partial and they must
    # NOT be summed:  dL/dmu[k] = (x[k] - mu[k])/sigma^2. Collapsing the vector
    # case into a single sum (which is what happened before `_gm_hyper_value`
    # kept vectors intact) put the whole sum into mu[1] and zero elsewhere.
    if ps.hyper_loc !== nothing
        if mu isa AbstractArray
            _gm_add_hyper_vec!(g, ps.hyper_loc, v, mu, s2, vals, layout)
        else
            acc = zero(T)
            @inbounds for x in v; acc += (x - mu); end
            _gm_add_hyper!(g, ps.hyper_loc, acc/s2, vals, layout)
        end
    end
    # hyper-scale: dL/dsigma = sum((x-mu)^2/sigma^2 - 1)/sigma. The scale is
    # always scalar (the recognizer only accepts scalar scale forms), so this
    # sum is correct in both cases.
    if ps.hyper_scale !== nothing
        ss = zero(T)
        @inbounds for k in eachindex(v); ss += (v[k] - _gm_mu_at(mu, k))^2; end
        dsd = (ss/s2 - _gm_c(T, length(v)))/sd
        _gm_add_hyper!(g, ps.hyper_scale, dsd, vals, layout)
    end
    return nothing
end

# Add a derivative w.r.t. a hyperparameter's CONSTRAINED value into g, chained
# through that hyperparameter's own bijector. Note this ADDS: a hyperparameter
# also has its own prior contribution, and may be shared by several sites.
function _gm_add_hyper!(g, name::Symbol, dval, vals, layout)
    slot = getproperty(layout.slots, name)
    i = first(slot.range)
    d = _gm_exemplar_by_name(name, layout)
    if _gm_positive_support(d)
        g[i] += dval * _gm_scalar(vals[name])   # x = exp(z), dx/dz = x
    else
        g[i] += dval
    end
    return nothing
end

# Vector hyper-location: each element of  gets its OWN partial, written
# into its own slot position, rather than one summed value.
function _gm_add_hyper_vec!(g, name::Symbol, v, mu, s2, vals, layout)
    slot = getproperty(layout.slots, name)
    d = _gm_exemplar_by_name(name, layout)
    pos = _gm_positive_support(d)
    mv = vals[name]
    o = first(slot.range) - 1
    @inbounds for k in eachindex(v)
        dk = (v[k] - mu[k])/s2
        g[o + k] += pos ? dk * mv[k] : dk
    end
    return nothing
end

function _gm_exemplar_by_name(name::Symbol, layout)
    for rec in layout.meta
        rec.name === name && return rec.dist_exemplar
    end
    error("gradmode: no layout record for $(name)")
end

# Adds `dv` (a derivative w.r.t. the CONSTRAINED value) into g at `range`,
# chained through the bijector to unconstrained space, plus the derivative of
# the log-Jacobian term itself.
#
# For an unconstrained (identity) parameter this is just `g[range] += dv`.
# For a positive parameter under exp: x = exp(z), dx/dz = x, and the logjac
# term is `z`, whose derivative is 1.
function _gm_add_chained!(g, range, dv, v, ps, layout)
    if _gm_is_positive(ps, layout)
        vv = collect(v)
        for (k, i) in enumerate(range)
            g[i] += dv[k] * vv[k] + 1.0
        end
    else
        for (k, i) in enumerate(range)
            g[i] += dv[k]
        end
    end
    return nothing
end

# Whether this site's support is positive (so the linked representation is a
# log). Read off the Layout's exemplar rather than assumed from the dist name,
# so a Truncated/derived distribution is classified by its actual support.
function _gm_is_positive(ps::PriorSite, layout)
    d = _gm_exemplar(ps, layout)
    return _gm_positive_support(d)
end
# NOTE ON COST. This is called once per prior site on EVERY gradient
# evaluation, and `minimum`/`maximum` on a distribution is not free: for an
# `MvNormal` they build a full extrema vector, measured at 9.6-10.2 us and 144-
# 1728 bytes per call against ~0 for a scalar distribution. That was essentially
# the whole of GradMode's fixed ~18 us floor -- a 50x2 model, whose analytic
# gradient is trivial arithmetic, cost about the same as a 50x200 one.
#
# The `Union{}` guard is the fix: a multivariate distribution has vector
# support, so "positive support with an unbounded upper end" (the scalar
# log-transform case this predicate is asking about) does not apply, and the
# expensive path can be skipped outright. Scalar distributions keep the original
# behaviour, which is already free.
#
# Kept as a computed predicate rather than a cached field because it reads the
# Layout's exemplar, so a Truncated or otherwise derived distribution is
# classified by its ACTUAL support instead of its name -- that is deliberate and
# worth preserving (see `_gm_is_positive`).
_gm_positive_support(d::Distributions.MultivariateDistribution) = false
_gm_positive_support(d::Distributions.MatrixDistribution) = false
_gm_positive_support(d) = try
    lo = minimum(d)
    lo >= 0 && isfinite(lo) && !isfinite(maximum(d))
catch
    false
end

# --- linear terms -----------------------------------------------------------

# Add one term's contribution into eta.
function _gm_accum_term!(eta, t::LinTerm, vals, args)
    if t.kind === :intercept
        a = _gm_scalar(vals[t.param])
        @inbounds @simd for i in eachindex(eta); eta[i] += a; end
    elseif t.kind === :matvec
        X = getfield(args, t.data)
        b = vals[t.param]
        LinearAlgebra.mul!(eta, X, b, true, true)   # eta .+= X*b
    elseif t.kind === :scaled
        x = getfield(args, t.data)
        a = _gm_scalar(vals[t.param])
        @inbounds @simd for i in eachindex(eta); eta[i] += a*x[i]; end
    elseif t.kind === :indexed
        v = vals[t.param]
        idx = getfield(args, t.index)
        @inbounds for i in eachindex(eta); eta[i] += v[idx[i]]; end
    elseif t.kind === :data
        x = getfield(args, t.data)
        @inbounds @simd for i in eachindex(eta); eta[i] += x[i]; end
    else
        error("gradmode: unknown term kind $(t.kind)")
    end
    return nothing
end

# Push dL/dEta back to this term's parameter. This is the transpose of
# `_gm_accum_term!` and is where the reverse-matvec (the whole point) happens.
function _gm_pullback_term!(g, t::LinTerm, w, vals, args, layout)
    t.param === nothing && return nothing     # :data has no parameter
    slot = getproperty(layout.slots, t.param)
    r = slot.range
    if t.kind === :intercept
        s = zero(eltype(w)); @inbounds @simd for i in eachindex(w); s += w[i]; end
        g[first(r)] += s
    elseif t.kind === :matvec
        X = getfield(args, t.data)
        LinearAlgebra.mul!(view(g, r), transpose(X), w, true, true)  # g[r] .+= X'w
    elseif t.kind === :scaled
        x = getfield(args, t.data)
        s = zero(eltype(w)); @inbounds @simd for i in eachindex(w); s += x[i]*w[i]; end
        g[first(r)] += s
    elseif t.kind === :indexed
        idx = getfield(args, t.index)
        o = first(r) - 1
        @inbounds for i in eachindex(w); g[o + idx[i]] += w[i]; end
    end
    return nothing
end

# --- links ------------------------------------------------------------------

"""
    _gm_link_loglik_and_dEta!(w, link, eta, y, sigma) -> loglik

Compute the log-likelihood and write `dL/dEta` into `w`. This is the ONLY
link-specific code in the whole gradient: everything downstream depends on
the term structure, not on the link.

  BernoulliLogit  dL/dEta = y - logistic(eta)
  LogPoisson      dL/dEta = y - exp(eta)
  Normal/MvNormal dL/dEta = (y - eta)/sigma^2
"""
function _gm_link_loglik_and_dEta!(w, link::Symbol, eta, y, sigma)
    T = eltype(eta)
    ll = zero(T)
    if link === :BernoulliLogit
        @inbounds @simd for i in eachindex(eta)
            p = _gm_logistic(eta[i])
            yi = y[i] > 0
            ll += yi ? log(p) : log1p(-p)
            w[i] = (yi ? one(T) : zero(T)) - p
        end
    elseif link === :LogPoisson
        # The -log(y!) normaliser is a constant w.r.t. theta, so it does not
        # affect the gradient — but it DOES affect the value, and the value is
        # compared against a general-AD reference in `check_gradmode`. It is
        # added once, outside the hot loop, via `_gm_logfactorial_sum` (which
        # needs no SpecialFunctions dependency: see its own comment).
        @inbounds @simd for i in eachindex(eta)
            m = exp(eta[i])
            yi = _gm_c(T, y[i])
            ll += yi*eta[i] - m
            w[i] = yi - m
        end
        ll -= _gm_c(T, _gm_logfactorial_sum(y))
    elseif link === :BernoulliCLogLog
        # complementary log-log: p = 1 - exp(-exp(eta)).
        # With m = exp(eta):  log p = log1p(-exp(-m)),  log(1-p) = -m.
        # dL/deta = y * m*exp(-m)/p  -  (1-y)*m
        # The y=0 branch is exactly -m, which is why this is written as two
        # branches rather than via a shared `p` — it avoids both a cancellation
        # and a division when y=0.
        @inbounds for i in eachindex(eta)
            m = exp(eta[i])
            if y[i] > 0
                em = exp(-m)
                p = -expm1(-m)          # 1 - exp(-m), accurate for small m
                ll += log(p)
                w[i] = m*em/p
            else
                ll += -m
                w[i] = -m
            end
        end
    elseif link === :Normal || link === :MvNormal
        # `sigma` may arrive as Float64 (from `_gm_obs_scale`'s AST-derived
        # default) even when eta is Float32; converting here keeps the loop at
        # one width instead of promoting every iteration.
        sd = _gm_c(T, sigma)
        s2 = sd*sd
        half = _gm_c(T, 0.5)
        @inbounds @simd for i in eachindex(eta)
            r = _gm_c(T, y[i]) - eta[i]
            ll += -half*r*r/s2
            w[i] = r/s2
        end
        ll += -_gm_c(T, length(eta))*(log(sd) + half*_gm_log2pi(T))
    else
        error("gradmode: no closed-form link derivative for $(link)")
    end
    return ll
end

@inline _gm_logistic(x) = inv(one(x) + exp(-x))

# sum(log(y_i!)) for integer counts. Constant w.r.t. theta (it only shifts the
# value, never the gradient), but it is still recomputed on every call, so its
# cost is not irrelevant. It used to sum log(j) for j = 2..y_i, making a call
# O(sum(y)): with counts in the millions (a Poisson GLM with a large linear
# predictor) GradMode came out 80-180x SLOWER than general AD. `_dl_logfactorial`
# (tilde.jl) keeps the cost per element constant.
function _gm_logfactorial_sum(y)
    s = 0.0
    @inbounds for v in y
        s += _dl_logfactorial(Float64, Int(v))
    end
    return s
end

# --- observation scale (the `sigma` of a Normal likelihood) -----------------

# The scale parameter of a Normal/MvNormal observe, if the model has one.
# Recognized models write `MvNormal(pred, sigma^2 * I)`; the scale is whichever
# prior-bound parameter appears in that argument.
function _gm_obs_scale(plan::GLMPlan, vals)
    plan.link in (:Normal, :MvNormal) || return 1.0
    plan.scale === nothing && return 1.0
    return _gm_scalar(vals[plan.scale])
end

# d(loglik)/d(sigma) = sum(r^2)/sigma^3 - N/sigma, chained through the log
# bijector (sigma is positive, so the linked value is log sigma and
# d/d(log sigma) = sigma * d/d(sigma)).
#
# `i` is the scale's theta index, from `gradmode_cache`. `sv` arrives untyped out
# of `vals`; taking it as an argument makes this a function barrier, so the loop
# below compiles for its concrete type instead of dispatching per operation
# (it was 0.42 us of a 3.9 us small-model gradient).
function _gm_scale_grad!(g, i::Int, sv, eta, y)
    T = _gm_T(eta)
    s = _gm_c(T, _gm_scalar(sv))
    ss = zero(T)
    @inbounds @simd for k in eachindex(eta)
        r = _gm_c(T, y[k]) - eta[k]; ss += r*r
    end
    dsigma = ss/(s^3) - _gm_c(T, length(eta))/s
    g[i] += dsigma * s     # chain through exp
    return nothing
end

# --- small helpers ----------------------------------------------------------

@inline _gm_scalar(v) = v isa AbstractArray ? v[1] : v

# The isotropic variance of an MvNormal prior site (1.0 when unspecified).
function _gm_mvnormal_var_of(ps::PriorSite)::Float64
    length(ps.args) >= 2 || return 1.0
    v = _gm_mvnormal_var(ps.args[2])
    v === nothing && error("gradmode: unrecognized MvNormal covariance for $(ps.name)")
    return v
end

# Numeric literal from a prior's argument list, or a default when absent.
#
# Sees through numeric CASTS (`pT(10)`, `Float32(2.5)`) and `zero(pT)`/`one(pT)`,
# which is how this package's own models are written (parameters are
# Float32-first, so distribution literals are routinely cast). Treating
# `pT(10)` as unparseable and silently falling back to the DEFAULT was a real
# bug: `Normal(zero(pT), pT(10))` was evaluated with sd=1 instead of 10, giving
# a wrong log-density and a wrong prior gradient. Caught by check_gradmode.
#
# Returns `nothing` when the argument is present but not a recognizable
# constant, so callers can distinguish "absent, use default" from "present but
# not understood" — the latter must reject rather than guess.
# The `::Float64` return annotation is load-bearing, not decoration.
# `PriorSite.args` is a `Vector{Any}` of unevaluated AST, so `_gm_const`'s walk
# below necessarily infers `Any` — and without this annotation that `Any`
# propagates into every arithmetic operation downstream (broadcasts, `sum`,
# `literal_pow`, `log`), leaving them as unresolved dynamic calls. This is the
# right place to put the boundary: it is exactly where AST walking ends and
# floating-point arithmetic begins, so annotating here concretises all the
# callers while the dynamic walk beneath stays dynamic.
#
# Measured effect (reported by the trim work on a gm_normal GLM): `_gm_num`
# accounted for 200 of 282 `--trim=safe` verifier errors, always as `::Any`.
function _gm_num(args, i, default)::Float64
    length(args) >= i || return default
    v = _gm_const(args[i])
    v === nothing && error("gradmode: non-constant prior argument $(args[i])")
    return v
end

# A compile-time numeric constant, seeing through casts and zero/one.
function _gm_const(a)::Union{Float64,Nothing}
    a isa Number && return float(a)
    if a isa Expr && a.head == :call && length(a.args) == 2
        f = _gm_head(a.args[1])
        f === :zero && return 0.0
        f === :one && return 1.0
        # a numeric cast such as `pT(10)` / `Float64(2.5)`
        a.args[2] isa Number && return float(a.args[2])
    end
    return nothing
end

# =============================================================================
# VERIFICATION
# =============================================================================

"""
    check_gradmode(plan, layout, args, ldf; n=8, rtol=1e-6, atol=1e-8, rng, verbose=false) -> Bool

Compare the closed-form gradient with the AD gradient of `ldf` at `n` random
points and return `true` if they agree. `plan` is
`PracticalBayes.gradmode_plan(model.f)`, `args` is `model.args`, and `ldf` is
built with `closed_form=false`.

```julia
ldf = LogDensityFunction(m, layout, store, AutoForwardDiff(); θ0=θ0, closed_form=false)
check_gradmode(PracticalBayes.gradmode_plan(m.f), layout, m.args, ldf)
```
"""
function check_gradmode(plan::GLMPlan, layout, args, ldf;
                        n::Int=8, rtol=1e-6, atol=1e-8, rng=Random.default_rng(), verbose::Bool=false)
    N = length(getfield(args, plan.response))
    ws = GradModeWorkspace{Float64}(N)
    g = zeros(Float64, layout.dim)
    ok = true
    for k in 1:n
        theta = randn(rng, layout.dim) .* 0.5
        v_ref, g_ref = LogDensityProblems.logdensity_and_gradient(ldf, theta)
        v_an = gradmode_value_and_gradient!(plan, layout, args, theta, g, ws)
        dv = abs(v_an - v_ref)
        dg = maximum(abs, g .- g_ref)
        good = isapprox(v_an, v_ref; rtol=rtol, atol=atol) &&
               isapprox(g, g_ref; rtol=rtol, atol=atol)
        good || (ok = false)
        verbose && println("  point $k: dvalue=$(dv)  maxdgrad=$(dg)  ", good ? "OK" : "MISMATCH")
    end
    return ok
end
