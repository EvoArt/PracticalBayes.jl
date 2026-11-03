# Tests for indexed OBSERVE sites: `y[i] ~ dist` where `y` is a model argument
# (compiler.jl `_tilde_expansion`, tilde.jl `tilde_index_observe`).
#
# This is the looped likelihood Turing users write by default. Before it was
# supported, every indexed site was treated as a parameter: the data became 20
# extra dimensions, tracing wrote prior draws over the caller's array, and
# sampling crashed storing a `Dual` into the `Vector{Float64}` data argument.

using Distributions: Normal, Exponential, Poisson, logpdf
# `NUTS` is PracticalBayes' own export (src/nuts.jl), not AdvancedHMC's type.
using ADTypes: AutoForwardDiff
using StableRNGs: StableRNG
import AbstractMCMC
import LogDensityProblems
using Statistics: mean, std

@model function loop_obs(y)
    mu ~ Normal(0, 1)
    for i in eachindex(y)
        y[i] ~ Normal(mu, 1)
    end
end

@model function dot_obs(y)
    mu ~ Normal(0, 1)
    y .~ Normal.(mu, 1)
end

@testset "indexed_observe.jl: looped observe is data, not parameters" begin
    y = fill(3.0, 20)
    y_before = copy(y)
    layout, θ0, _ = build_layout(loop_obs(y))

    @test keys(layout.slots) == (:mu,)
    @test layout.dim == 1
    @test y == y_before                      # tracing no longer overwrites the data

    # Same log density and gradient as the vectorized form, at several points.
    # `closed_form=false` so both sides really go through the model body.
    ldl = LogDensityFunction(loop_obs(y), layout, NamedTuple(), AutoForwardDiff(); θ0=θ0, closed_form=false)
    lay_d, θd, _ = build_layout(dot_obs(y))
    ldd = LogDensityFunction(dot_obs(y), lay_d, NamedTuple(), AutoForwardDiff(); θ0=θd, closed_form=false)
    for θ in ([0.0], [1.7], [-2.3])
        vl, gl = LogDensityProblems.logdensity_and_gradient(ldl, θ)
        vd, gd = LogDensityProblems.logdensity_and_gradient(ldd, θ)
        @test vl ≈ vd
        @test gl ≈ gd
    end
    @test y == y_before
end

@testset "indexed_observe.jl: observe path does not allocate" begin
    # The EvalMode method is one element read plus one logpdf, the same work
    # as the scalar observe — it must not add per-element allocation.
    y = randn(StableRNG(1), 200)
    layout, θ0, _ = build_layout(loop_obs(y))
    ldf = LogDensityFunction(loop_obs(y), layout, NamedTuple(), nothing)
    LogDensityProblems.logdensity(ldf, θ0)   # compile
    @test (@allocated LogDensityProblems.logdensity(ldf, θ0)) == 0
end

@testset "indexed_observe.jl: NUTS on a looped observe matches the analytic posterior" begin
    y = fill(3.0, 20)
    chn = AbstractMCMC.sample(StableRNG(1), loop_obs(y), NUTS(0.8), 1000;
                              n_adapts=500, discard_initial=500, progress=false)
    mu = vec(chn[:mu])
    post_var = 1 / (1 + length(y))
    @test abs(mean(mu) - post_var * sum(y)) < 5 * std(mu) / sqrt(length(mu))
    @test y == fill(3.0, 20)
end

@testset "indexed_observe.jl: alongside an indexed parameter family" begin
    # A per-group effect family (assume, local container) next to a looped
    # observe over an argument: only the argument becomes data.
    @model function groups(y, g, k)
        tau ~ Exponential(1)
        b = Vector{paramtype(__mode__)}(undef, k)
        for j in 1:k
            b[j] ~ Normal(0, tau)
        end
        for i in eachindex(y)
            y[i] ~ Normal(b[g[i]], 1)
        end
    end
    y = [0.1, 0.4, 2.0, 2.2, -1.0]
    g = [1, 1, 2, 2, 3]
    m = groups(y, g, 3)
    layout, θ0, _ = build_layout(m)
    @test keys(layout.slots) == (:tau, :b)
    @test layout.dim == 4

    # log density against a hand-written reference at one point
    ldf = LogDensityFunction(m, layout, NamedTuple(), nothing)
    θ = [log(0.7), 0.2, 1.9, -0.8]          # tau is log-linked; b is identity
    tau, b = 0.7, θ[2:4]
    ref = logpdf(Exponential(1), tau) + log(tau) +
          sum(logpdf(Normal(0, tau), bj) for bj in b) +
          sum(logpdf(Normal(b[g[i]], 1), y[i]) for i in eachindex(y))
    @test LogDensityProblems.logdensity(ldf, θ) ≈ ref
end

@testset "indexed_observe.jl: matrix data and `:=` containers" begin
    @model function mat_obs(Y)
        mu ~ Normal(0, 1)
        for i in axes(Y, 1), j in axes(Y, 2)
            Y[i, j] ~ Normal(mu, 1)
        end
    end
    Y = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    layout, θ0, _ = build_layout(mat_obs(Y))
    @test layout.dim == 1
    ldf = LogDensityFunction(mat_obs(Y), layout, NamedTuple(), nothing)
    @test LogDensityProblems.logdensity(ldf, [0.5]) ≈
          logpdf(Normal(0, 1), 0.5) + sum(logpdf.(Normal(0.5, 1), Y))

    # a `:=` name is "already has a value", exactly like an argument
    @model function walrus_obs(x)
        mu ~ Normal(0, 1)
        z := 2 .* x
        for i in eachindex(z)
            z[i] ~ Normal(mu, 1)
        end
    end
    x = [0.5, 1.0]
    layout, _, _ = build_layout(walrus_obs(x))
    @test layout.dim == 1
    ldf = LogDensityFunction(walrus_obs(x), layout, NamedTuple(), nothing)
    @test LogDensityProblems.logdensity(ldf, [0.3]) ≈
          logpdf(Normal(0, 1), 0.3) + sum(logpdf.(Normal(0.3, 1), 2 .* x))
end

@testset "indexed_observe.jl: missing elements" begin
    # density-evaluating paths need data at every element
    y = [1.0, missing, 3.0]
    @test_throws ArgumentError build_layout(loop_obs(y))
    @test_throws ArgumentError logjoint(loop_obs(y), (; mu=0.0))
    @test_throws ArgumentError pointwise_loglikelihoods(loop_obs(y), (; mu=0.0))
end

@testset "indexed_observe.jl: rand, predict and pointwise log-likelihoods" begin
    n = 4
    m_pred = loop_obs(fill(missing, n))

    # prior predictive: y comes back as ONE plain vector, as with `.~`
    d = rand(StableRNG(2), m_pred)
    @test d.y isa Vector{Float64}
    @test length(d.y) == n

    # the caller's missing-filled array is never written into
    ymiss = fill(missing, n)
    rand(StableRNG(2), loop_obs(ymiss))
    @test all(ismissing, ymiss)

    # posterior predictive: one y vector per draw, centred on that draw's mu
    draws = [(; mu=10.0), (; mu=-10.0)]
    preds = predict(StableRNG(3), m_pred, draws)
    @test length(preds) == 2
    @test all(p -> p.y isa Vector{Float64} && length(p.y) == n, preds)
    @test all(>(5), preds[1].y) && all(<(-5), preds[2].y)

    # a partially observed array keeps its data and draws only the missing slot
    p = predict(StableRNG(4), loop_obs([1.0, missing, 3.0]), [(; mu=0.0)])[1]
    @test p.y[1] == 1.0 && p.y[3] == 3.0 && !ismissing(p.y[2])

    # pointwise: one value per element, summing to loglikelihood_at, and
    # identical to what `.~` reports for the same data
    y = [0.2, -0.4, 1.1]
    pw = pointwise_loglikelihoods(loop_obs(y), (; mu=0.3))
    @test pw.y ≈ logpdf.(Normal(0.3, 1), y)
    @test sum(pw.y) ≈ loglikelihood_at(loop_obs(y), (; mu=0.3))
    @test pw.y ≈ pointwise_loglikelihoods(dot_obs(y), (; mu=0.3)).y
end
