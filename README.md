# PracticalBayes

[![CI](https://github.com/EvoArt/PracticalBayes.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/EvoArt/PracticalBayes.jl/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://evoart.github.io/PracticalBayes.jl/)

PracticalBayes is a probabilistic programming package for Julia, and a work in
progress. It is built largely from the building blocks of the Turing ecosystem,
with its own model evaluation and some changes so that it works well with the (also WIP) PracticallyBayesian packages. Models are written with Turing-style `@model` and
`~` syntax and sampled with HMC/NUTS, alone or as blocks of a Gibbs sampler.

```julia
using Pkg
Pkg.add("PracticalBayes")
```

```julia
using PracticalBayes, Distributions

@model function linreg(x, y)
    α ~ Normal(0, 10)
    β ~ Normal(0, 10)
    σ ~ Exponential(1)
    y .~ Normal.(α .+ β .* x, σ)
end

x = range(0, 1; length=50)
y = 1 .+ 2 .* x .+ 0.3 .* randn(50)
chn = sample(linreg(x, y), NUTS(), 1000)
chn[:β]
```

## Features

- **Float32.** Models can run in Float32 end to end.
- **Trimmable models.** A simple model can be compiled into a standalone binary
  with `juliac --trim`.
- **GPU data.** Data can live on a CUDA device. This is untested on GPU
  hardware and a work in progress.
- **Closed-form gradients** for models recognised as generalised linear models,
  used automatically.

See the [documentation](https://evoart.github.io/PracticalBayes.jl/).

## Relationship to Turing

PracticalBayes is assembled from Turing ecosystem packages (AbstractPPL,
AdvancedHMC, Bijectors, Distributions, LogDensityProblems, FlexiChains). It has
its own model evaluation in place of DynamicPPL, does not depend on Turing
itself, and covers a narrower range of models. Most Turing models, and most of the
[Turing tutorials](https://turinglang.org/docs/tutorials/), run with small changes; see
[Port a model from Turing](https://evoart.github.io/PracticalBayes.jl/howto/from_turing/).

<!-- BENCH:START -->
Ratios are median gradient time `PracticalBayes / Turing` (`< 1` means PracticalBayes faster). Corner cells come from the 5x5 sweep grid: N in [50, 200, 1000, 5000, 20000], NPARAMS in [2, 10, 50, 100, 200]. Turing evaluates in `Float64` whatever the input type, so the `Float32` rows compare PracticalBayes in `Float32` against Turing in `Float64`.

Measured with Julia 1.12.7 on one Julia thread and 1 BLAS thread of an exclusive Intel(R) Xeon(R) Silver 4216 CPU @ 2.10GHz node: PracticalBayes 0.1.0, Turing 0.49.0, DynamicPPL 0.42.14, ForwardDiff 1.4.7, Mooncake 0.5.67, Enzyme 0.13.216, Piste 0.3.0 (003fd5d), AdvancedHMC 0.8.8, Distributions 0.25.133, BenchmarkTools 1.8.0. Each figure is the median of up to 200 samples or 3 s, with the evaluations per sample tuned by BenchmarkTools.

### ForwardDiff

| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |
|---|---:|---:|---:|---:|---:|
| `normal` | `Float64` | 1.058 | 1.027 | 0.983 | 1.002 |
| `normal` | `Float32` | 0.996 | 1.03 | 1.068 | 1.005 |
| `poisson` | `Float64` | 1.102 | 1.024 | 0.961 | 0.981 |
| `poisson` | `Float32` | 0.894 | 1.53 | 0.938 | 1.009 |
| `bernoulli_logit` | `Float64` | 0.974 | 1.062 | 0.983 | 0.989 |
| `bernoulli_logit` | `Float32` | 0.987 | 1.537 | 1.0 | 1.0 |

### Mooncake

| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |
|---|---:|---:|---:|---:|---:|
| `normal` | `Float64` | 1.141 | 0.88 | 0.405 | 1.002 |
| `normal` | `Float32` | 1.001 | 0.897 | 1.399 | 0.943 |
| `poisson` | `Float64` | 0.761 | 0.849 | 0.785 | 0.975 |
| `poisson` | `Float32` | 0.705 | 0.864 | 0.849 | 0.896 |
| `bernoulli_logit` | `Float64` | 0.981 | 0.882 | 1.13 | 1.002 |
| `bernoulli_logit` | `Float32` | 0.932 | 0.614 | 0.955 | 0.971 |

### Enzyme

| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |
|---|---:|---:|---:|---:|---:|
| `normal` | `Float64` | 0.86 | 1.068 | 0.913 | 0.991 |
| `normal` | `Float32` | 1.178 | 0.899 | 1.016 | 1.013 |
| `poisson` | `Float64` | 0.763 | 0.949 | 0.892 | 0.983 |
| `poisson` | `Float32` | 0.585 | 0.946 | 1.009 | 0.997 |
| `bernoulli_logit` | `Float64` | 1.101 | 0.779 | 0.991 | 0.999 |
| `bernoulli_logit` | `Float32` | 1.367 | 0.988 | 0.97 | 0.996 |


### Piste (PracticalBayes' own AD)

[Piste](https://github.com/EvoArt/Piste.jl) is the only backend here that survives `juliac --trim`, i.e. the only one that lets a model compile to a standalone binary. It is not yet available through DifferentiationInterface, which is how Turing selects an AD backend, so the comparison here is between backends rather than between packages: `Piste / fastest of ForwardDiff, Mooncake, Enzyme` on the same PracticalBayes model, with `< 1` meaning Piste is faster.

Read the tables together. **Forward mode** costs O(P) by construction: with few parameters it is level with the fastest backend on the Poisson and Bernoulli models and a few times slower on the normal one, and with many parameters it falls far behind, which is what the large-P columns show. **Reverse mode** does not grow with P the same way. On the Poisson and Bernoulli models it is within about 2x of the fastest backend everywhere, and ahead of it when there are many observations and many parameters. On the normal model it is slower, most of all with many observations and few parameters.

**GradMode** is not AD at all: for a GLM-shaped model PracticalBayes recognises the shape and evaluates the analytic gradient directly (`X' * resid`, one BLAS call, no tape). On the Poisson and Bernoulli models it is the fastest option in every cell. On the normal model it is level with or faster than the AD backends from about a thousand observations up; below that, where a gradient takes a few microseconds, its fixed cost per call can make it up to about 2x slower. All three sweep models are GLM-shaped, so it applies to every cell; a model outside the recognised grammar falls back to ordinary AD and gets none of this.

#### Forward mode

| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |
|---|---:|---:|---:|---:|---:|
| `normal` | `Float64` | 1.535 | 54.613 | 3.114 | 94.458 |
| `normal` | `Float32` | 1.769 | 50.978 | 3.722 | 120.268 |
| `poisson` | `Float64` | 1.088 | 28.187 | 1.012 | 30.131 |
| `poisson` | `Float32` | 1.078 | 42.15 | 0.992 | 35.701 |
| `bernoulli_logit` | `Float64` | 1.213 | 34.35 | 0.96 | 33.652 |
| `bernoulli_logit` | `Float32` | 0.89 | 42.389 | 0.92 | 36.65 |


#### Reverse mode

| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |
|---|---:|---:|---:|---:|---:|
| `normal` | `Float64` | 3.978 | 1.794 | 7.474 | 1.423 |
| `normal` | `Float32` | 2.747 | 2.176 | 15.814 | 2.221 |
| `poisson` | `Float64` | 1.422 | 0.953 | 1.656 | 0.668 |
| `poisson` | `Float32` | 1.671 | 1.92 | 1.687 | 0.824 |
| `bernoulli_logit` | `Float64` | 1.801 | 1.209 | 1.835 | 0.714 |
| `bernoulli_logit` | `Float32` | 1.804 | 1.941 | 1.738 | 0.888 |


#### GradMode (analytic, GLM-shaped models only)

| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |
|---|---:|---:|---:|---:|---:|
| `normal` | `Float64` | 1.782 | 0.941 | 0.391 | 0.973 |
| `normal` | `Float32` | 1.223 | 0.517 | 0.359 | 0.919 |
| `poisson` | `Float64` | 0.448 | 0.481 | 0.347 | 0.418 |
| `poisson` | `Float32` | 0.536 | 0.475 | 0.361 | 0.377 |
| `bernoulli_logit` | `Float64` | 0.658 | 0.585 | 0.448 | 0.43 |
| `bernoulli_logit` | `Float32` | 0.547 | 0.477 | 0.436 | 0.392 |

<!-- BENCH:END -->
