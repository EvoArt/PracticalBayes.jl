# Regression tests for the specialized `_dot_loglik` kernels in src/tilde.jl.
#
# These kernels replace `Distributions.loglikelihood` on the `.~` hot path with
# hand-written SIMD loops. They are a pure optimization: every one MUST return
# the same value `Distributions` does, INCLUDING the out-of-support `-Inf`
# cases, or a wrong posterior is the result. That equivalence is what these
# tests pin down — the speedup itself is measured by bench/dot_loglik_oneoff.jl
# (a manual, one-time check, deliberately not part of CI).
import Distributions
import StableRNGs

# Allocation of one warm call, measured inside a function: written directly in
# a testset body, `@allocated` on Julia 1.10 also counts boxing the call's
# Float64 result (16 bytes), which is not the code under test.
_dl_allocs(f, args...) = (f(args...); @allocated f(args...))

@testset "_dot_loglik specialized kernels" begin
    dl = PracticalBayes._dot_loglik
    rng = StableRNGs.StableRNG(20250904)

    y_n = randn(rng, 500)
    y_p = Float64.(rand(rng, Distributions.Poisson(3.0), 500))
    y_b = Float64.(rand(rng, Distributions.Bernoulli(0.35), 500))
    y_e = rand(rng, Distributions.Exponential(2.0), 500)

    @testset "matches Distributions.loglikelihood" begin
        for (d, y) in ((Distributions.Normal(0.5, 1.3), y_n),
                       (Distributions.Normal(-2.0, 0.25), y_n),
                       (Distributions.Exponential(2.0), y_e),
                       (Distributions.Poisson(3.0), y_p),
                       (Distributions.Poisson(0.05), y_p),
                       (Distributions.Bernoulli(0.35), y_b))
            @test dl(d, y) ≈ Distributions.loglikelihood(d, y) rtol=1e-10
        end
    end

    @testset "large Poisson counts: correct, and cost independent of the count" begin
        # log(k!) was a sum over 2..k, so a log density cost O(sum(y)). A
        # Poisson GLM with a large linear predictor has counts in the millions.
        lf = PracticalBayes._dl_logfactorial
        for k in (0, 1, 2, 15, 16, 17, 40, 10^6)
            @test lf(Float64, k) ≈ Distributions.logfactorial(k) rtol=1e-12
        end
        @test lf(Float32, 10^6) isa Float32
        y_big = Float64.(rand(rng, Distributions.Poisson(2.0e6), 200))
        d_big = Distributions.Poisson(2.0e6)
        # Looser than the rtol=1e-10 above: the kernel forms sum(y)*log(lambda)
        # and sum(log y!) separately, each ~6e9 here, and subtracts them to get
        # ~-1750, so ~1e-7 of absolute rounding is expected and harmless.
        @test dl(d_big, y_big) ≈ Distributions.loglikelihood(d_big, y_big) rtol=1e-8
        # GradMode's normaliser goes through the same helper.
        @test PracticalBayes._gm_logfactorial_sum(y_big) ≈
              sum(Distributions.logfactorial.(Int.(y_big))) rtol=1e-12
        # A count of 10^6 must cost about what a count of 20 does, not 10^5x.
        dl(d_big, y_big); dl(Distributions.Poisson(20.0), fill(20.0, 200))
        t_big = minimum(@elapsed(dl(d_big, y_big)) for _ in 1:20)
        t_small = minimum(@elapsed(dl(Distributions.Poisson(20.0), fill(20.0, 200))) for _ in 1:20)
        @test t_big < 50 * t_small + 1e-5
    end

    @testset "out-of-support data gives -Inf, not a wrong finite value" begin
        # The whole risk of a hand-written kernel is that it happily computes
        # a plausible number for data the distribution gives zero density to.
        @test dl(Distributions.Exponential(2.0), vcat(y_e, -1.0)) == -Inf
        @test dl(Distributions.Poisson(3.0), vcat(y_p, -1.0)) == -Inf
        @test dl(Distributions.Poisson(3.0), vcat(y_p, 2.5)) == -Inf   # non-integer
        @test dl(Distributions.Bernoulli(0.35), vcat(y_b, 2.0)) == -Inf
    end

    @testset "boundary parameter values" begin
        # 0*log(0) is NaN if computed naively; the true log-likelihood is 0.
        @test dl(Distributions.Bernoulli(0.0), zeros(10)) == 0.0
        @test dl(Distributions.Bernoulli(1.0), ones(10)) == 0.0
        @test dl(Distributions.Bernoulli(0.0), vcat(zeros(9), 1.0)) == -Inf
        @test dl(Distributions.Bernoulli(1.0), vcat(ones(9), 0.0)) == -Inf
        # degenerate scale falls back to Distributions rather than dividing by 0
        @test dl(Distributions.Normal(0.0, 0.0), y_n) ==
              Distributions.loglikelihood(Distributions.Normal(0.0, 0.0), y_n)
        @test dl(Distributions.Normal(0.5, 1.3), [0.2]) ≈
              Distributions.logpdf(Distributions.Normal(0.5, 1.3), 0.2)
    end

    @testset "Float32 stays Float32" begin
        # The Float32-first parameter path: a Float32 distribution and Float32
        # data must not silently widen the accumulator to Float64. Tolerance is
        # loose because both this kernel and Distributions accumulate in
        # Float32 and drift differently (see the note in src/tilde.jl).
        d32 = Distributions.Normal(0.5f0, 1.3f0)
        v32 = dl(d32, Float32.(y_n))
        @test v32 isa Float32
        @test v32 ≈ Distributions.loglikelihood(d32, Float32.(y_n)) rtol=1e-2
    end

    @testset "unspecialized distributions still work" begin
        # Anything without a kernel must fall through to the generic methods.
        d = Distributions.TDist(4.0)
        @test dl(d, y_n) ≈ Distributions.loglikelihood(d, y_n) rtol=1e-10
        # array of DIFFERENT distributions: the lazy-broadcast-sum fallback
        ds = Distributions.Normal.(randn(rng, 50), 1.2)
        yy = randn(rng, 50)
        @test dl(ds, yy) ≈ sum(Distributions.logpdf.(ds, yy)) rtol=1e-10
        # ... which never materializes the per-observation log-densities
        # The claim is "no per-observation temporary", i.e. allocation that
        # does not grow with the data: it is 0 on Julia 1.11+, and a fixed 16
        # bytes on 1.10 whatever the length.
        a_small = _dl_allocs(dl, ds, yy)
        a_big = _dl_allocs(dl, Distributions.Normal.(randn(rng, 5000), 1.2), randn(rng, 5000))
        @test a_small == a_big
        @test a_small <= 16
        # ... and keeps broadcasting's shape rules: one distribution per ROW
        # of a matrix, and a mismatch is still an error, not a silent wrong sum
        Y = randn(rng, 50, 3)
        @test dl(ds, Y) ≈ sum(Distributions.logpdf.(ds, Y)) rtol=1e-10
        @test_throws DimensionMismatch dl(ds, randn(rng, 49))
    end

    @testset "end-to-end through a `.~` model" begin
        # The kernels are only useful if they are what `.~` actually calls.
        PracticalBayes.@model function _dl_model(x, y)
            mu ~ Distributions.Normal(0.0, 1.0)
            y .~ Distributions.Normal.(mu .+ x, 0.7)
        end
        x = randn(rng, 200)
        y = 1.5 .+ x .+ randn(rng, 200) .* 0.7
        m = _dl_model(x, y)
        # logjoint at a fixed point must match a hand-computed reference
        ref = Distributions.logpdf(Distributions.Normal(0.0, 1.0), 0.3) +
              sum(Distributions.logpdf.(Distributions.Normal.(0.3 .+ x, 0.7), y))
        @test PracticalBayes.logjoint(m, (; mu = 0.3)) ≈ ref rtol=1e-10
    end
end
