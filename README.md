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

Measured with Julia 1.12.7 on one Julia thread and 1 BLAS thread of an exclusive Intel(R) Xeon(R) Silver 4216 CPU @ 2.10GHz node: PracticalBayes 0.1.0, Turing 0.49.0, DynamicPPL 0.42.14, ForwardDiff 1.4.7, Mooncake 0.5.67, Enzyme 0.13.216, Piste 0.3.0 (92b011b), AdvancedHMC 0.8.8, Distributions 0.25.133, BenchmarkTools 1.8.0. Each figure is the median of up to 200 samples or 3 s, with the evaluations per sample tuned by BenchmarkTools.

### ForwardDiff

| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |
|---|---:|---:|---:|---:|---:|
| `normal` | `Float64` | 1.051 | 1.009 | 1.23 | 0.994 |
| `normal` | `Float32` | 0.993 | 1.035 | 1.004 | 1.004 |
| `poisson` | `Float64` | 1.116 | 1.047 | 0.967 | 0.968 |
| `poisson` | `Float32` | 1.183 | 1.515 | 0.952 | 1.009 |
| `bernoulli_logit` | `Float64` | 1.032 | 1.016 | 1.029 | 0.988 |
| `bernoulli_logit` | `Float32` | 1.094 | 1.573 | 0.998 | 0.991 |

### Mooncake

| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |
|---|---:|---:|---:|---:|---:|
| `normal` | `Float64` | 1.171 | 0.93 | 0.769 | 1.295 |
| `normal` | `Float32` | 1.008 | 1.185 | 1.404 | 0.94 |
| `poisson` | `Float64` | 0.914 | 0.849 | 0.797 | 1.064 |
| `poisson` | `Float32` | 0.736 | 0.895 | 0.751 | 0.91 |
| `bernoulli_logit` | `Float64` | 1.089 | 0.89 | 1.091 | 1.005 |
| `bernoulli_logit` | `Float32` | 0.894 | 0.818 | 1.083 | 0.972 |

### Enzyme

| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |
|---|---:|---:|---:|---:|---:|
| `normal` | `Float64` | 0.784 | 0.933 | 1.0 | 1.002 |
| `normal` | `Float32` | 0.844 | 1.287 | 1.003 | 1.033 |
| `poisson` | `Float64` | 0.601 | 0.95 | 0.858 | 0.976 |
| `poisson` | `Float32` | 0.579 | 1.043 | 1.05 | 1.002 |
| `bernoulli_logit` | `Float64` | 0.953 | 0.95 | 1.434 | 1.008 |
| `bernoulli_logit` | `Float32` | 1.366 | 0.957 | 0.97 | 0.996 |


### Piste (PracticalBayes' own AD)

[Piste](https://github.com/EvoArt/Piste.jl) is the only backend here that survives `juliac --trim`, i.e. the only one that lets a model compile to a standalone binary. It is not yet available through DifferentiationInterface, which is how Turing selects an AD backend, so the comparison here is between backends rather than between packages: `Piste / fastest of ForwardDiff, Mooncake, Enzyme` on the same PracticalBayes model, with `< 1` meaning Piste is faster.

Read the tables together. **Forward mode** costs O(P) by construction: with few parameters it is level with the fastest backend on the Poisson and Bernoulli models and a few times slower on the normal one, and with many parameters it falls far behind, which is what the large-P columns show. **Reverse mode** does not grow with P the same way. On the Poisson and Bernoulli models it is within about 1.8x of the fastest backend everywhere, and ahead of it when there are many parameters: with many observations at both precisions, and with few observations in `Float64`. On the normal model it is slower, about 1.3-3x in `Float64` and up to 7x in `Float32`, most of all with many observations and few parameters.

**GradMode** is not AD at all: for a GLM-shaped model PracticalBayes recognises the shape and evaluates the analytic gradient directly (`X' * resid`, one BLAS call, no tape). On the Poisson and Bernoulli models it is the fastest option in every cell. On the normal model it is level with or faster than the AD backends from about a thousand observations up (in this run the single largest `Float64` cell came out 1.2x slower); below that, where a gradient takes a few microseconds, its fixed cost per call can make it up to about 2x slower. All three sweep models are GLM-shaped, so it applies to every cell; a model outside the recognised grammar falls back to ordinary AD and gets none of this.

#### Forward mode

| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |
|---|---:|---:|---:|---:|---:|
| `normal` | `Float64` | 1.451 | 63.806 | 2.92 | 86.998 |
| `normal` | `Float32` | 2.419 | 51.426 | 3.65 | 115.258 |
| `poisson` | `Float64` | 0.784 | 28.47 | 1.006 | 34.064 |
| `poisson` | `Float32` | 1.052 | 38.07 | 1.001 | 36.252 |
| `bernoulli_logit` | `Float64` | 0.88 | 29.222 | 0.89 | 32.774 |
| `bernoulli_logit` | `Float32` | 0.803 | 42.529 | 0.92 | 35.918 |


#### Reverse mode

| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |
|---|---:|---:|---:|---:|---:|
| `normal` | `Float64` | 2.643 | 1.358 | 3.174 | 1.27 |
| `normal` | `Float32` | 2.406 | 1.486 | 7.203 | 1.719 |
| `poisson` | `Float64` | 1.359 | 0.757 | 1.577 | 0.603 |
| `poisson` | `Float32` | 1.528 | 1.272 | 1.633 | 0.776 |
| `bernoulli_logit` | `Float64` | 1.625 | 0.767 | 1.548 | 0.676 |
| `bernoulli_logit` | `Float32` | 1.155 | 1.721 | 1.626 | 0.883 |


#### GradMode (analytic, GLM-shaped models only)

| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |
|---|---:|---:|---:|---:|---:|
| `normal` | `Float64` | 1.817 | 1.1 | 0.371 | 1.214 |
| `normal` | `Float32` | 1.64 | 0.522 | 0.346 | 0.876 |
| `poisson` | `Float64` | 0.468 | 0.48 | 0.346 | 0.418 |
| `poisson` | `Float32` | 0.53 | 0.434 | 0.365 | 0.382 |
| `bernoulli_logit` | `Float64` | 0.648 | 0.493 | 0.408 | 0.406 |
| `bernoulli_logit` | `Float32` | 0.488 | 0.478 | 0.444 | 0.386 |

<!-- BENCH:END -->
