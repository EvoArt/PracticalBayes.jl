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

Measured with Julia 1.12.7 on one Julia thread and 1 BLAS thread of an exclusive Intel(R) Xeon(R) Silver 4216 CPU @ 2.10GHz node: PracticalBayes 0.1.0, Turing 0.49.0, DynamicPPL 0.42.14, ForwardDiff 1.4.7, Mooncake 0.5.67, Enzyme 0.13.216, Piste 0.3.0, AdvancedHMC 0.8.8, Distributions 0.25.133, BenchmarkTools 1.8.0. Each figure is the median of up to 200 samples or 3 s, with the evaluations per sample tuned by BenchmarkTools.

### ForwardDiff

| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |
|---|---:|---:|---:|---:|---:|
| `normal` | `Float64` | 1.043 | 1.031 | 1.002 | 0.964 |
| `normal` | `Float32` | 2.118 | 1.026 | 1.003 | 0.985 |
| `poisson` | `Float64` | 1.126 | 1.016 | 0.97 | 1.014 |
| `poisson` | `Float32` | 0.927 | 1.516 | 1.037 | 1.006 |
| `bernoulli_logit` | `Float64` | 0.995 | 1.023 | 0.999 | 1.009 |
| `bernoulli_logit` | `Float32` | 0.991 | 1.528 | 0.998 | 1.0 |

### Mooncake

| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |
|---|---:|---:|---:|---:|---:|
| `normal` | `Float64` | 1.132 | 0.891 | 0.453 | 1.009 |
| `normal` | `Float32` | 0.707 | 0.894 | 0.865 | 0.981 |
| `poisson` | `Float64` | 0.76 | 0.849 | 0.801 | 0.968 |
| `poisson` | `Float32` | 0.733 | 0.827 | 0.724 | 0.921 |
| `bernoulli_logit` | `Float64` | 0.891 | 0.904 | 1.051 | 1.007 |
| `bernoulli_logit` | `Float32` | 0.929 | 0.87 | 1.006 | 0.949 |

### Enzyme

| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |
|---|---:|---:|---:|---:|---:|
| `normal` | `Float64` | 0.852 | 0.97 | 0.928 | 0.996 |
| `normal` | `Float32` | 0.772 | 0.641 | 0.989 | 0.938 |
| `poisson` | `Float64` | 1.099 | 1.049 | 0.881 | 0.987 |
| `poisson` | `Float32` | 1.425 | 0.95 | 0.995 | 0.989 |
| `bernoulli_logit` | `Float64` | 0.96 | 0.788 | 1.021 | 0.999 |
| `bernoulli_logit` | `Float32` | 0.957 | 1.002 | 1.026 | 1.021 |


### Piste (PracticalBayes' own AD)

[Piste](https://github.com/EvoArt/Piste.jl) is the only backend here that survives `juliac --trim`, i.e. the only one that lets a model compile to a standalone binary. Turing cannot use it, so there is no PB/Turing ratio to give; these ratios are instead `Piste / fastest of ForwardDiff, Mooncake, Enzyme` on the same cell, with `< 1` meaning Piste is faster.

Read the tables together. **Forward mode** costs O(P) by construction: with few parameters it is level with the fastest backend, and with many it falls far behind, which is what the large-P columns show. **Reverse mode** does not grow with P the same way. In `Float64` it is close to the fastest backend on the Poisson and Bernoulli models and ahead of it at large P, but several times slower on the normal model; in `Float32` it is slower throughout.

**GradMode** is not AD at all: for a GLM-shaped model PracticalBayes recognises the shape and evaluates the analytic gradient directly (`X' * resid`, one BLAS call, no tape). On the Poisson and Bernoulli models it is the fastest option almost everywhere. On the normal model it is level with the AD backends from about a thousand observations up, and slower below that, where a gradient takes microseconds and GradMode's fixed cost per call dominates. All three sweep models are GLM-shaped, so it applies to every cell; a model outside the recognised grammar falls back to ordinary AD and gets none of this.

#### Forward mode

| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |
|---|---:|---:|---:|---:|---:|
| `normal` | `Float64` | 1.364 | 61.807 | 3.015 | 94.039 |
| `normal` | `Float32` | 2.367 | 70.717 | 3.863 | 119.986 |
| `poisson` | `Float64` | 0.706 | 27.307 | 1.011 | 33.894 |
| `poisson` | `Float32` | 1.003 | 41.843 | 0.948 | 37.415 |
| `bernoulli_logit` | `Float64` | 0.903 | 29.428 | 0.932 | 32.687 |
| `bernoulli_logit` | `Float32` | 1.004 | 40.352 | 0.917 | 35.616 |


#### Reverse mode

| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |
|---|---:|---:|---:|---:|---:|
| `normal` | `Float64` | 2.878 | 2.035 | 7.384 | 1.436 |
| `normal` | `Float32` | 2.891 | 4.494 | 16.769 | 9.344 |
| `poisson` | `Float64` | 1.105 | 0.869 | 1.668 | 0.677 |
| `poisson` | `Float32` | 1.683 | 2.682 | 1.643 | 3.626 |
| `bernoulli_logit` | `Float64` | 1.37 | 1.025 | 1.771 | 0.662 |
| `bernoulli_logit` | `Float32` | 1.939 | 2.712 | 1.77 | 3.44 |


#### GradMode (analytic, GLM-shaped models only)

| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |
|---|---:|---:|---:|---:|---:|
| `normal` | `Float64` | 4.869 | 1.711 | 0.422 | 0.979 |
| `normal` | `Float32` | 5.176 | 1.208 | 0.416 | 0.926 |
| `poisson` | `Float64` | 0.835 | 0.511 | 0.343 | 0.42 |
| `poisson` | `Float32` | 0.81 | 0.656 | 0.345 | 0.396 |
| `bernoulli_logit` | `Float64` | 0.77 | 0.68 | 0.433 | 0.399 |
| `bernoulli_logit` | `Float32` | 1.028 | 0.558 | 0.441 | 0.377 |

<!-- BENCH:END -->
