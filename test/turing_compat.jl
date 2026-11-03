# Tests for the Turing-compatible API surface: PB's own `NUTS` (src/nuts.jl),
# `setprogress!`, `decondition(model)`, chain-level `predict`/`returned`,
# calling a model, and type arguments to `@model`. These are what let Turing
# tutorial code run with `using PracticalBayes` in place of `using Turing`.
#
# NB: no testset local may share a name with a model variable — a `@model`
# defined in a testset is a closure and would assign it (see CLAUDE.md).

using Distributions: Normal, Exponential, MvNormal
using StableRNGs: StableRNG
using Statistics: mean
using LinearAlgebra: I
import AbstractMCMC
import AdvancedHMC
import FlexiChains
import LogDensityProblems
using ADTypes: AutoForwardDiff

@testset "turing_compat.jl: NUTS constructor" begin
    # Turing's default target acceptance, and AdvancedHMC's own sampler back.
    @test NUTS() isa AdvancedHMC.NUTS
    @test NUTS().δ == 0.65
    @test NUTS(0.8) isa AdvancedHMC.NUTS
    @test NUTS(0.8f0).δ isa Float32
    @test NUTS(0.8; max_depth=5).max_depth == 5
    # `adtype` rides along on the sampler.
    s = NUTS(; adtype=AutoForwardDiff())
    @test s isa PracticalBayes._ADSampler
    @test s.sampler.δ == 0.65

    @model function tc_simple(y)
        mu ~ Normal(0, 1)
        y .~ Normal.(mu, 1.0)
    end
    y = randn(StableRNG(1), 20)
    chn = sample(StableRNG(2), tc_simple(y), NUTS(; adtype=AutoForwardDiff()), 200; progress=false)
    @test chn isa FlexiChains.SymChain
    @test size(chn, 1) == 200
    # and inside Gibbs, per block
    @model function tc_two(y)
        mu ~ Normal(0, 1)
        sigma ~ Exponential(1)
        y .~ Normal.(mu, sigma)
    end
    spl = Gibbs(:mu => NUTS(; adtype=AutoForwardDiff()), :sigma => NUTS())
    g = sample(StableRNG(3), tc_two(y), spl, 50; n_adapts=20, progress=false, chain_type=nothing)
    @test length(g) == 50
end

@testset "turing_compat.jl: setprogress!" begin
    try
        @test setprogress!(false) === false
        @test default_progress() === false
        @test setprogress!(true) === true
        @test default_progress() === true
    finally
        PracticalBayes._PROGRESS_OVERRIDE[] = nothing   # leave the session as found
    end
end

@testset "turing_compat.jl: decondition, predict and returned on chains" begin
    rng = StableRNG(4)
    yobs = randn(rng, 15) .+ 3.0

    @model function tc_pred(n)
        mu ~ Normal(0, 5)
        y ~ MvNormal(fill(mu, n), I)
        return (; twice=2mu)
    end
    m = tc_pred(length(yobs)) | (; y=yobs)
    @test haskey(m.conditioned, :y)
    @test isempty(decondition(m).conditioned)

    chn = sample(StableRNG(5), m, NUTS(0.8), 300; progress=false)
    pred = predict(StableRNG(6), decondition(m), chn)
    @test pred isa FlexiChains.SymChain
    @test FlexiChains.parameters(pred) == [:y]          # only what was predicted
    ydraws = vec(pred[:y])
    @test length(ydraws) == 300
    @test all(length(d) == length(yobs) for d in ydraws)
    # the predictive mean follows the posterior of mu (~3)
    @test abs(mean(mean.(ydraws)) - mean(vec(chn[:mu]))) < 0.2
    @test Set(FlexiChains.parameters(predict(StableRNG(6), decondition(m), chn; include_all=true))) == Set([:mu, :y])

    rv = returned(m, chn)
    @test size(rv) == (300, 1)
    @test rv[1].twice ≈ 2 * vec(chn[:mu])[1]

    # calling a model returns its return value, parameters drawn from the prior
    @test m(StableRNG(7)) isa NamedTuple{(:twice,)}
end

@testset "turing_compat.jl: type arguments to @model" begin
    # Turing's idiom: a container allocated from a type ARGUMENT. Under AD the
    # argument must become the AD number type, or `z[i] ~ ...` cannot store.
    @model function tc_typed(y, ::Type{T}=Vector{Float64}) where {T}
        z = T(undef, 2)
        for i in 1:2
            z[i] ~ Normal(0, 1)
        end
        y .~ Normal.(z[1] + z[2], 1.0)
    end
    y = randn(StableRNG(8), 10)
    m = tc_typed(y)
    layout, θ0, store = build_layout(m)
    ldf = LogDensityFunction(m, layout, store, AutoForwardDiff(); θ0=θ0)
    v, g = LogDensityProblems.logdensity_and_gradient(ldf, θ0)
    @test isfinite(v)
    @test length(g) == 2 && all(isfinite, g)
    chn = sample(StableRNG(9), m, NUTS(0.8), 100; progress=false)
    @test size(chn, 1) == 100

    # a type parameter bound by a `Val` argument (previously unsupported)
    @model function tc_val(y, ::Val{k}) where {k}
        w ~ MvNormal(zeros(k), I)
        y .~ Normal.(sum(w), 1.0)
    end
    m2 = tc_val(y, Val(3))
    layout2, θ2, store2 = build_layout(m2)
    @test length(θ2) == 3
    @test isfinite(LogDensityProblems.logdensity(LogDensityFunction(m2, layout2, store2), θ2))
end

@testset "turing_compat.jl: MCMCThreads with progress on" begin
    # AbstractMCMC gives each chain one of ITS progress objects (not a Bool)
    # under MCMCThreads; PB's own sampling loops used to MethodError on it.
    # Tutorials hit this because AbstractMCMC's progress default is on.
    @model function tc_threads(y)
        mu ~ Normal(0, 1)
        sigma ~ Exponential(1)
        y .~ Normal.(mu, sigma)
    end
    y = randn(StableRNG(30), 12)
    c1 = sample(StableRNG(31), tc_threads(y), NUTS(0.8), MCMCThreads(), 20, 2; progress=true)
    @test size(c1)[1:2] == (20, 2)
    spl = Gibbs(:mu => NUTS(0.8), :sigma => NUTS(0.8))
    c2 = sample(StableRNG(32), tc_threads(y), spl, MCMCThreads(), 20, 2; n_adapts=10, progress=true)
    @test size(c2)[1:2] == (20, 2)
end

@testset "turing_compat.jl: logprior / loglikelihood / logjoint in the chain" begin
    # Turing chains carry these per draw (constrained scale, no Jacobian), and
    # tutorials read `chn[:logjoint]`. They must equal PB's own point-wise
    # `logjoint(model, draw)`, not AdvancedHMC's unconstrained `log_density`.
    @model function tc_lp(y)
        sigma ~ Exponential(1)
        y .~ Normal.(0.0, sigma)
    end
    y = randn(StableRNG(40), 10)
    m = tc_lp(y)
    chn = sample(StableRNG(41), m, NUTS(0.8), 20; progress=false)
    lj = vec(chn[:logjoint])
    s = vec(chn[:sigma])
    @test length(lj) == 20
    @test lj[3] ≈ logjoint(m, (; sigma=s[3]))
    @test vec(chn[:logprior])[3] + vec(chn[:loglikelihood])[3] ≈ lj[3]
end
