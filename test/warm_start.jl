# Tests for src/warm_start.jl and AdaptiveHMC's `adapt` mode.
#
# The target is Gaussian with a known covariance, so the carried metric is
# checked against an absolute value rather than against another run:
#   a ~ N(0, 1),  b | a ~ N(a, 0.5^2),  y | b ~ N(b, 1),  y = 0.5
# has precision [5 -4; -4 5], covariance [5 4; 4 5] / 9, mean [2, 2.5] / 9.

using Distributions: Normal
using AdvancedHMC: NUTS
using StableRNGs: StableRNG
import AbstractMCMC
using Statistics: mean, cov
using Test
using PracticalBayes

@model function _ws_gauss(y)
    a ~ Normal(0, 1)
    b ~ Normal(a, 0.5)
    y ~ Normal(b, 1)
end

const _WS_COV = [5.0 4.0; 4.0 5.0] ./ 9
const _WS_MEAN = [2.0, 2.5] ./ 9

function _ws_run(spl, n; n_adapts, init=(; a=0.0, b=0.0), seed=1)
    m = _ws_gauss(0.5)
    rng = StableRNG(seed)
    t, st = AbstractMCMC.step(rng, m, spl; init=init, n_adapts=n_adapts)
    draws = [[t.a, t.b]]
    for _ in 2:n
        t, st = AbstractMCMC.step(rng, m, spl, st; n_adapts=n_adapts)
        push!(draws, [t.a, t.b])
    end
    return reduce(hcat, draws)', st
end

@testset "warm_start: tuning is read, carried and kept" begin
    # One block holding every parameter of a GLM-shaped model: the case that
    # takes the closed-form gradient inside Gibbs.
    spl = Gibbs((:a, :b) => AdaptiveHMC(0.8; n_leapfrog=8, metric=:dense))
    _, st = _ws_run(spl, 1500; n_adapts=1000)
    tu = only(hmc_tuning(spl, st))
    @test tu.names == (:a, :b)
    @test tu.metric isa Matrix{Float64}
    # Stan's regularised windowed estimate of the posterior covariance.
    @test tu.metric ≈ _WS_COV atol=0.12
    @test tu.step_size > 0

    # n_adapts = 0: used exactly as carried, however long the run.
    warm = warm_start(spl, hmc_tuning(spl, st))
    x, st0 = _ws_run(warm, 4000; n_adapts=0, init=st.values, seed=2)
    tu0 = only(hmc_tuning(warm, st0))
    @test tu0.metric == tu.metric
    @test tu0.step_size == tu.step_size
    @test vec(mean(x; dims=1)) ≈ _WS_MEAN atol=0.05
    @test cov(x) ≈ _WS_COV atol=0.1

    # adapt = :step_size: the metric is kept, only the step size moves.
    ws = warm_start(spl, hmc_tuning(spl, st); adapt=:step_size)
    _, st1 = _ws_run(ws, 300; n_adapts=200, init=st.values, seed=3)
    tu1 = only(hmc_tuning(ws, st1))
    @test tu1.metric == tu.metric
    @test tu1.step_size != tu.step_size
    @test tu1.step_size > 0

    # adapt = :full: the metric is re-learnt.
    wf = warm_start(spl, hmc_tuning(spl, st); adapt=:full)
    _, st2 = _ws_run(wf, 300; n_adapts=200, init=st.values, seed=4)
    @test only(hmc_tuning(wf, st2)).metric != tu.metric
end

@testset "warm_start: diagonal metric round-trips as a vector" begin
    spl = Gibbs(:a => AdaptiveHMC(0.8; n_leapfrog=8), :b => AdaptiveHMC(0.8; n_leapfrog=8))
    _, st = _ws_run(spl, 600; n_adapts=400)
    tu = hmc_tuning(spl, st)
    @test all(t -> t.metric isa Vector{Float64} && length(t.metric) == 1, tu)
    warm = warm_start(spl, tu)
    _, st0 = _ws_run(warm, 50; n_adapts=0, init=st.values, seed=5)
    @test [t.metric for t in hmc_tuning(warm, st0)] == [t.metric for t in tu]
end

@testset "warm_start: refuses what it cannot honour" begin
    spl = Gibbs((:a, :b) => AdaptiveHMC(0.8; n_leapfrog=8, metric=:dense))
    _, st = _ws_run(spl, 50; n_adapts=40)
    tu = hmc_tuning(spl, st)
    @test_throws ArgumentError warm_start(spl, tu; adapt=:metric)
    @test_throws ArgumentError warm_start(Gibbs((:b, :a) => AdaptiveHMC(0.8)), tu)
    @test_throws ArgumentError warm_start(
        Gibbs(:a => AdaptiveHMC(0.8), :b => AdaptiveHMC(0.8)), tu)
    @test_throws ArgumentError warm_start(Gibbs((:a, :b) => NUTS(0.8)), tu; adapt=:step_size)
    @test warm_start(Gibbs((:a, :b) => NUTS(0.8)), tu; adapt=:full) isa Gibbs
    fixed = Gibbs((:a, :b) => HMC(8))
    @test only(warm_start(fixed, tu).blocks).kernel === only(fixed.blocks).kernel
    @test_throws ArgumentError AdaptiveHMC(0.8; adapt=:metric)
end
