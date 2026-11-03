# Tests for src/particle_gibbs.jl — the `PG` (conditional SMC) Gibbs kernel.
#
# Exactness is checked against posteriors computed by ENUMERATING every latent
# trajectory, so a wrong weight, a wrong resampling step, or a reference
# particle that is not held fixed shows up as a biased marginal rather than
# merely a slow chain.

using Distributions: Normal, Categorical, Bernoulli, Dirichlet, MvNormal, logpdf
# `NUTS` is PracticalBayes' own export (src/nuts.jl), not AdvancedHMC's type.
using StableRNGs: StableRNG
using Statistics: mean
using LinearAlgebra: I

@testset "particle_gibbs.jl: Markov-chain latent — exact marginals" begin
    # A 2-state Markov chain observed with noise. T=5 gives 32 trajectories to
    # enumerate, and the states are dependent, so this exercises the
    # stage-by-stage resampling rather than independent per-site updates.
    T = 5
    y = [0.1, 1.9, 2.3, -0.4, 2.1]
    P = [0.8 0.2; 0.3 0.7]
    mus = (0.0, 2.0)

    @model function pg_chain(y, P, mus)
        n = length(y)
        z = Vector{Int}(undef, n)
        z[1] ~ Categorical([0.5, 0.5])
        y[1] ~ Normal(mus[z[1]], 1.0)
        for t in 2:n
            z[t] ~ Categorical(P[z[t - 1], :])
            y[t] ~ Normal(mus[z[t]], 1.0)
        end
    end

    # Exact posterior P(z_t = 2 | y) by brute force.
    logw = Float64[]
    trajs = Vector{Vector{Int}}()
    for bits in 0:(2^T - 1)
        z = [((bits >> (t - 1)) & 1) + 1 for t in 1:T]
        lw = log(0.5) + sum(log(P[z[t - 1], z[t]]) for t in 2:T) +
             sum(logpdf(Normal(mus[z[t]], 1.0), y[t]) for t in 1:T)
        push!(logw, lw)
        push!(trajs, z)
    end
    w = exp.(logw .- maximum(logw))
    w ./= sum(w)
    exact = [sum(w[i] for i in eachindex(trajs) if trajs[i][t] == 2) for t in 1:T]

    draws = sample(StableRNG(1), pg_chain(y, P, mus), Gibbs(:z => PG(10)), 6_000;
                   discard_initial=200, chain_type=nothing, progress=false)
    est = [mean(d.z[t] == 2 for d in draws) for t in 1:T]
    # Binomial MC-SE at 6000 draws is <= 0.0065; PG draws are autocorrelated,
    # so allow a few times that.
    @test maximum(abs.(est .- exact)) < 0.03
end

@testset "particle_gibbs.jl: @addlogprob! terms enter the weights" begin
    # The only evidence about `z` comes through `@addlogprob!`, which bypasses
    # every tilde method. If PG missed it the draws would follow the prior (0.5).
    @model function pg_addlogprob(yobs)
        z ~ Bernoulli(0.5)
        @addlogprob! logpdf(Normal(z ? 2.0 : 0.0, 1.0), yobs)
    end
    yobs = 1.6
    l1 = logpdf(Normal(2.0, 1.0), yobs)
    l0 = logpdf(Normal(0.0, 1.0), yobs)
    exact = exp(l1) / (exp(l0) + exp(l1))

    draws = sample(StableRNG(2), pg_addlogprob(yobs), Gibbs(:z => PG(5)), 6_000;
                   chain_type=nothing, progress=false)
    @test abs(mean(d.z for d in draws) - exact) < 0.03
end

@testset "particle_gibbs.jl: PG + NUTS Gibbs on a mixture" begin
    # The Gaussian-mixture tutorial's shape: continuous means/weights under
    # NUTS, discrete assignments under PG. Well-separated clusters, so the
    # posterior cluster locations are close to the generating values — up to
    # LABELS: a mixture's labels are not identified, and with clusters this far
    # apart a Gibbs chain stays in whichever labelling it started in (seen: a
    # start drawn from a different RNG stream converged to the swapped one).
    # So the checks below relabel the draws to the ordering of `truth` first.
    rng = StableRNG(7)
    N = 30
    truth = [-2.0, 2.0]
    ztrue = rand(rng, 1:2, N)
    x = [truth[ztrue[i]] + 0.5 * randn(rng) for i in 1:N]

    @model function pg_mixture(x)
        mu ~ MvNormal([-1.0, 1.0], I)
        k = Vector{Int}(undef, length(x))
        for i in eachindex(x)
            k[i] ~ Categorical([0.5, 0.5])
            x[i] ~ Normal(mu[k[i]], 0.5)
        end
    end

    spl = Gibbs(:k => PG(20), :mu => NUTS(0.8))
    draws = sample(StableRNG(3), pg_mixture(x), spl, 400; n_adapts=100, discard_initial=100,
                   chain_type=nothing, progress=false)
    # `mu_hat`, not `mu`: the model above is a closure over this testset's
    # locals, so a local named after a model variable would be assigned by it.
    mu_hat = mean(d.mu for d in draws)
    kbar = mean(d.k for d in draws)
    if mu_hat[1] > mu_hat[2]          # the swapped labelling: relabel 1 <-> 2
        mu_hat = reverse(mu_hat)
        kbar = 3 .- kbar
    end
    @test abs(mu_hat[1] - truth[1]) < 0.4
    @test abs(mu_hat[2] - truth[2]) < 0.4
    # Assignments recovered for nearly every point.
    @test mean(abs.(kbar .- ztrue) .< 0.5) > 0.9
end

@testset "particle_gibbs.jl: argument checks" begin
    @test_throws ArgumentError PG(1)
end
