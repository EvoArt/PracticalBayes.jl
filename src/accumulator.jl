"""
    Accum{T<:Real}

The log density accumulated while a model runs, kept as two fields:
`logprior` (including the log-Jacobian of any transform) and `loglik`.

`Accum(0.0)` starts both at zero. Read the totals with [`logprior`](@ref),
[`loglikelihood_`](@ref) and [`logjoint`](@ref).
"""
struct Accum{T<:Real}
    logprior::T
    loglik::T
end

Accum(z) = Accum(z, z)
Accum{T}() where {T<:Real} = Accum{T}(zero(T), zero(T))
# Explicit promoting constructor: `struct Accum{T}`'s implicit default
# constructor requires BOTH fields to already be the same concrete `T`, but
# `acc_prior`/`acc_lik` below routinely combine values of different types
# (e.g. accumulating a Float64 `logpdf` result into a Float32-seeded `Accum`,
# which happens whenever a model mixes distributions/constants of different
# precisions during TraceMode, or under mixed-precision Dual arithmetic).
# Promoting here means the accumulator's type naturally widens instead of
# throwing a MethodError on the first mismatched site.
Accum(logprior, loglik) = Accum(promote(logprior, loglik)...)

@inline acc_prior(a::Accum, x) = Accum(a.logprior + x, a.loglik)
@inline acc_lik(a::Accum, x) = Accum(a.logprior, a.loglik + x)

"""
    logprior(a::Accum)

The log-prior accumulated in `a`.
"""
@inline logprior(a::Accum) = a.logprior

# Named with a trailing underscore (not `loglikelihood`) to avoid colliding
# with the `loglikelihood` generic function widely exported by
# StatsAPI/Distributions/etc — a user doing `using PracticalBayes, Distributions`
# would otherwise hit an ambiguous-name error merely from loading both.
"""
    loglikelihood_(a::Accum)

The log-likelihood accumulated in `a`. The trailing underscore avoids a clash
with `Distributions.loglikelihood`. For a model at a point, use
[`loglikelihood_at`](@ref).
"""
@inline loglikelihood_(a::Accum) = a.loglik

"""
    logjoint(a::Accum)

`logprior(a) + loglikelihood_(a)`.
"""
@inline logjoint(a::Accum) = a.logprior + a.loglik

Base.eltype(::Accum{T}) where {T} = T
Base.zero(::Type{Accum{T}}) where {T} = Accum{T}()
