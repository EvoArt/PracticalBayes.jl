# PracticalBayes' own `NUTS` constructor.
#
# PB used to re-export AdvancedHMC's `NUTS` directly. Turing code writes two
# forms that type does not accept: `NUTS()` (Turing defaults the target
# acceptance rate to 0.65; AdvancedHMC requires it) and
# `NUTS(; adtype=AutoMooncake())` (Turing carries the AD backend on the sampler;
# AdvancedHMC's NUTS has no such field, and PB takes it as a `sample` keyword).
# Adding those methods to `AdvancedHMC.NUTS` from here would be type piracy, so
# PB exports this function instead. With no `adtype` it returns exactly
# `AdvancedHMC.NUTS(δ; kwargs...)`, so everything that already worked is
# unchanged; with one, it wraps the sampler in `_ADSampler`, which `sample` and
# `Gibbs` unwrap.

"""
    NUTS(δ=0.65; adtype=nothing, max_depth=10, Δ_max=1000.0, integrator=:leapfrog, metric=:diagonal)

The No-U-Turn sampler with target acceptance rate `δ` (default 0.65, Turing's
default). All keywords except `adtype` are AdvancedHMC's, and without `adtype`
the result is AdvancedHMC's own `NUTS` sampler.

`adtype` chooses the AD backend for this sampler, as in Turing
(`NUTS(; adtype=AutoMooncake())`). It is equivalent to passing `adtype=` to
`sample`; inside [`Gibbs`](@ref) it sets the backend for that block alone.

A Float32 `δ` (`NUTS(0.8f0)`) makes `sample` run in Float32.
"""
function NUTS(δ::Real=0.65; adtype=nothing, kwargs...)
    spl = AdvancedHMC.NUTS(δ; kwargs...)
    return adtype === nothing ? spl : _ADSampler(spl, adtype)
end

# A sampler carrying its own AD backend. Deliberately an AbstractMCMC sampler
# (so MCMCThreads etc. route it through PB's `sample`) that is never stepped
# itself: `sample` and `Gibbs` take `sampler` out and pass `adtype` along.
struct _ADSampler{S<:AbstractMCMC.AbstractSampler,AD} <: AbstractMCMC.AbstractSampler
    sampler::S
    adtype::AD
end

# What Gibbs steps, and the AD backend a block uses: the block's own when its
# sampler carries one, else the `adtype` given to `sample`.
_block_sampler(k) = k
_block_sampler(k::_ADSampler) = k.sampler
_block_adtype(k, default) = default
_block_adtype(k::_ADSampler, _) = k.adtype
