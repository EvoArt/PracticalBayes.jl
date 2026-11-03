# Tests for the generic `sample` method (src/sample.jl) and generic Gibbs
# sampler blocks (src/gibbs.jl): any AbstractMCMC sampler that works on a
# `LogDensityModel` and implements `getparams`/`getstats`/`setparams!!`.
#
# The sampler used here is a minimal random-walk Metropolis written against
# that interface alone, rather than AdvancedMH: it proves the path needs
# nothing package-specific, and adds no test dependency.

using Distributions: Normal, Exponential
# `NUTS` is PracticalBayes' own export (src/nuts.jl), not AdvancedHMC's type.
using StableRNGs: StableRNG
import AbstractMCMC
import LogDensityProblems
using Statistics: mean, std
import FlexiChains

struct _TestRWMH <: AbstractMCMC.AbstractSampler
    scale::Float64
end

struct _TestRWState
    θ::Vector{Float64}
    lp::Float64
    accepted::Bool
end

AbstractMCMC.getparams(s::_TestRWState) = s.θ
AbstractMCMC.getstats(s::_TestRWState) = (lp=s.lp, accepted=s.accepted)
AbstractMCMC.setparams!!(m::AbstractMCMC.LogDensityModel, s::_TestRWState, θ) =
    _TestRWState(θ, LogDensityProblems.logdensity(m.logdensity, θ), s.accepted)

function AbstractMCMC.step(rng, m::AbstractMCMC.LogDensityModel, ::_TestRWMH; initial_params, kwargs...)
    s = _TestRWState(copy(initial_params), LogDensityProblems.logdensity(m.logdensity, initial_params), true)
    return s, s
end

function AbstractMCMC.step(rng, m::AbstractMCMC.LogDensityModel, spl::_TestRWMH, s::_TestRWState; kwargs...)
    θp = s.θ .+ spl.scale .* randn(rng, length(s.θ))
    lp = LogDensityProblems.logdensity(m.logdensity, θp)
    next = log(rand(rng)) < lp - s.lp ? _TestRWState(θp, lp, true) : _TestRWState(s.θ, s.lp, false)
    return next, next
end

@testset "samplers.jl: any AbstractMCMC sampler — conjugate Normal posterior" begin
    rng = StableRNG(42)
    y = randn(rng, 30) .+ 2.0

    @model function rw_conjugate(y)
        mu ~ Normal(0, 1)
        y .~ Normal.(mu, 1.0)
    end

    n = length(y)
    post_var = 1 / (1 + n)
    post_mean = post_var * sum(y)

    chn = sample(StableRNG(1), rw_conjugate(y), _TestRWMH(0.4), 20_000;
                 discard_initial=1_000, thinning=2, progress=false)
    @test chn isa FlexiChains.SymChain
    # Never name a local after a model variable here: a `@model` defined inside
    # a `@testset` is a closure, so its `mu ~ ...` would assign THIS local on
    # every later evaluation (a gradient call leaves a Dual in it).
    mu_draws = vec(chn[:mu])
    @test length(mu_draws) == 20_000
    # Random-walk draws are strongly autocorrelated, so the naive MC-SE
    # understates the error; 0.05 is ~5 of them at this posterior sd (0.18)
    # and chain length, and far below any systematic error worth catching.
    @test abs(mean(mu_draws) - post_mean) < 0.05
    @test abs(std(mu_draws) - sqrt(post_var)) < 0.03

    # `getstats` becomes the chain's Extra keys.
    extra = Set(FlexiChains.get_name(k) for k in keys(chn) if k isa FlexiChains.Extra)
    @test :accepted in extra
    @test :lp in extra

    # chain_type=nothing hands back the sampler's own transitions.
    raw = sample(StableRNG(1), rw_conjugate(y), _TestRWMH(0.4), 10; chain_type=nothing, progress=false)
    @test raw isa AbstractVector{_TestRWState}
end

@testset "samplers.jl: constrained parameters are mapped back" begin
    # sigma lives in (0, ∞); the sampler walks the unconstrained (log) scale,
    # and every draw in the chain must be on the constrained scale.
    rng = StableRNG(5)
    y = randn(rng, 40) .* 2.0

    @model function rw_scale(y)
        sigma ~ Exponential(1)
        y .~ Normal.(0.0, sigma)
    end

    chn = sample(StableRNG(2), rw_scale(y), _TestRWMH(0.3), 4_000; discard_initial=500, progress=false)
    s = vec(chn[:sigma])
    @test all(>(0), s)
    @test 1.5 < mean(s) < 2.6
end

@testset "samplers.jl: a generic sampler as a Gibbs block" begin
    rng = StableRNG(11)
    y = randn(rng, 30) .+ 1.0

    @model function rw_two(y)
        mu ~ Normal(0, 1)
        sigma ~ Exponential(1)
        y .~ Normal.(mu, sigma)
    end

    # One block per sampler family: the RW block's `setparams!!` must pick up
    # NUTS's latest `sigma`, or `mu` would be sampled from a stale conditional.
    spl = Gibbs(:mu => _TestRWMH(0.3), :sigma => NUTS(0.8))
    chn = sample(StableRNG(3), rw_two(y), spl, 3_000; n_adapts=200, discard_initial=500, progress=false)
    mu_draws = vec(chn[:mu])
    sigma_draws = vec(chn[:sigma])
    # Reference: the same model under plain NUTS.
    ref = sample(StableRNG(4), rw_two(y), NUTS(0.8), 3_000; n_adapts=500, discard_initial=500, progress=false)
    @test abs(mean(mu_draws) - mean(vec(ref[:mu]))) < 0.08
    @test abs(mean(sigma_draws) - mean(vec(ref[:sigma]))) < 0.08
end
