# PracticalBayes

[![CI](https://github.com/EvoArt/PracticalBayes/actions/workflows/ci.yml/badge.svg)](https://github.com/EvoArt/PracticalBayes/actions/workflows/ci.yml)
[![Benchmark Sweep](https://github.com/EvoArt/PracticalBayes/actions/workflows/benchmark.yml/badge.svg)](https://github.com/EvoArt/PracticalBayes/actions/workflows/benchmark.yml)
[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://EvoArt.github.io/PracticalBayes)

PracticalBayes is a probabilistic programming package for Julia, and a work in
progress. It is built largely from the building blocks of the Turing ecosystem,
with its own model evaluation and some changes so that it works well with the
PracticallyBayesian packages. Models are written with Turing-style `@model` and
`~` syntax and sampled with HMC/NUTS, alone or as blocks of a Gibbs sampler.

```julia
using Pkg
Pkg.add(url="https://github.com/EvoArt/PracticalBayes")
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

- **Latent kernels as plain functions.** A `Gibbs` block can be your own
  `AbstractLatentKernel`: one `latent_step` method that receives the model's
  data and the current value of every parameter, and returns new values for its
  block. It runs once per MCMC step, outside the gradient. It is similar to
  Turing's `GibbsConditional` and `externalsampler`, but designed to be easy to
  use with the latent-state samplers of epidemic models, such as those in
  [EpidemicTrajectories.jl](https://evoart.github.io/EpidemicTrajectories/dev/tutorials/cattle_iffbs/).
- **Float32.** Models can run in Float32 end to end.
- **Trimmable models.** A simple model can be compiled into a standalone binary
  with `juliac --trim`.
- **GPU data.** Data can live on a CUDA device. This is untested on GPU
  hardware and a work in progress.
- **Closed-form gradients** for models recognised as generalised linear models,
  used automatically.

See the [documentation](https://EvoArt.github.io/PracticalBayes) for a
tutorial, how-to guides and the API reference.

## Relationship to Turing

PracticalBayes is assembled from Turing ecosystem packages (AbstractPPL,
AdvancedHMC, Bijectors, Distributions, LogDensityProblems, FlexiChains). It has
its own model evaluation in place of DynamicPPL, does not depend on Turing
itself, and covers a narrower range of models. Most Turing models, and most of the
[Turing tutorials](https://turinglang.org/docs/tutorials/), run with small changes; see
[Port a model from Turing](https://EvoArt.github.io/PracticalBayes/howto/from_turing/).

## Benchmark summary (auto-updated)

Ratios below are median gradient time `PracticalBayes / Turing` (lower is better for PracticalBayes).

<!-- BENCH:START -->
Ratios are median gradient time `PracticalBayes / Turing` (`< 1` means PracticalBayes faster). Corner cells come from the 5x5 sweep grid: N in [50, 200, 1000, 5000, 20000], NPARAMS in [2, 10, 50, 100, 200].

### Forwarddiff

| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |
|---|---:|---:|---:|---:|---:|
| `normal` | `Float64` | 1.0 | 1.079 | 1.206 | 0.986 |
| `normal` | `Float32` | 0.9 | 1.009 | 1.052 | 0.992 |
| `poisson` | `Float64` | 0.46 | 1.146 | 0.574 | 1.094 |
| `poisson` | `Float32` | 0.483 | 0.914 | 0.504 | 0.893 |
| `bernoulli_logit` | `Float64` | 1.0 | 1.09 | 1.013 | 0.981 |
| `bernoulli_logit` | `Float32` | 0.971 | 1.011 | 0.959 | 1.005 |

### Mooncake

| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |
|---|---:|---:|---:|---:|---:|
| `normal` | `Float64` | 1.042 | 1.143 | 1.009 | 0.765 |
| `normal` | `Float32` | 1.196 | 1.107 | 1.608 | 1.164 |
| `poisson` | `Float64` | 1.512 | 1.418 | 0.677 | 1.014 |
| `poisson` | `Float32` | 0.687 | 0.942 | 0.737 | 0.874 |
| `bernoulli_logit` | `Float64` | 0.976 | 1.052 | 0.889 | 1.012 |
| `bernoulli_logit` | `Float32` | 0.959 | 0.596 | 0.998 | 1.107 |

### Enzyme

| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |
|---|---:|---:|---:|---:|---:|
| `normal` | `Float64` | 0.949 | 1.126 | 0.95 | 1.107 |
| `normal` | `Float32` | 0.658 | 0.89 | 0.909 | 1.008 |
| `poisson` | `Float64` | n/a | n/a | n/a | n/a |
| `poisson` | `Float32` | n/a | n/a | n/a | n/a |
| `bernoulli_logit` | `Float64` | n/a | n/a | n/a | n/a |
| `bernoulli_logit` | `Float32` | n/a | n/a | n/a | n/a |


### Piste (PracticalBayes' own AD)

[Piste](https://github.com/EvoArt/Piste.jl) is the only backend here that survives `juliac --trim`, i.e. the only one that lets a model compile to a standalone binary. Turing cannot use it, so there is no PB/Turing ratio to give; these ratios are instead `Piste / fastest of ForwardDiff, Mooncake, Enzyme` on the same cell, with `< 1` meaning Piste is faster.

Read the tables together. **Forward mode** wins at small problems (the other backends' per-call setup dominates a tiny gradient) and loses badly at large P, because its cost is O(P) by construction -- the large bottom-right figures are that, not a defect. **Reverse mode** is the one to use on parameter-heavy models; it costs one sweep regardless of P.

**GradMode** is not AD at all: for a GLM-shaped model PracticalBayes recognises the shape and evaluates the analytic gradient directly (`X' * resid`, one BLAS call, no tape), which is why it beats every backend here. All three sweep models are GLM-shaped, so it applies to every cell; a model outside the recognised grammar falls back to ordinary AD and gets none of this.

#### Forward mode

| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |
|---|---:|---:|---:|---:|---:|
| `normal` | `Float64` | 1.667 | 24.368 | 1.024 | 39.034 |
| `normal` | `Float32` | 1.667 | 33.113 | 1.415 | 19.13 |
| `poisson` | `Float64` | 1.828 | 12.19 | 0.997 | 30.242 |
| `poisson` | `Float32` | 1.034 | 9.757 | 1.03 | 19.317 |
| `bernoulli_logit` | `Float64` | 1.394 | 16.121 | 0.996 | 32.784 |
| `bernoulli_logit` | `Float32` | 0.794 | 23.365 | 0.906 | 12.446 |


#### Reverse mode

| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |
|---|---:|---:|---:|---:|---:|
| `normal` | `Float64` | 5.333 | 1.393 | 4.565 | 1.183 |
| `normal` | `Float32` | 5.667 | 3.558 | 9.625 | 2.822 |
| `poisson` | `Float64` | 2.5 | 0.661 | 2.101 | 1.041 |
| `poisson` | `Float32` | 2.69 | 1.428 | 2.59 | 2.261 |
| `bernoulli_logit` | `Float64` | 2.379 | 1.075 | 2.18 | 1.439 |
| `bernoulli_logit` | `Float32` | 2.529 | 3.345 | 3.005 | 2.354 |


#### GradMode (analytic, GLM-shaped models only)

Note: these figures predate a 3.8x speedup to GradMode's fixed per-call cost (`_gm_positive_support` was calling `minimum`/`maximum` on an `MvNormal` prior on every gradient, ~10us and allocating). Regenerate the sweep to pick it up; the small-N columns here are the ones most understated.

| Likelihood | Precision | small N / small P | small N / large P | large N / small P | large N / large P |
|---|---:|---:|---:|---:|---:|
| `normal` | `Float64` | 36.056 | 1.186 | 0.28 | 0.284 |
| `normal` | `Float32` | 41.389 | 2.093 | 0.72 | 0.204 |
| `poisson` | `Float64` | 10.862 | 39.628 | 0.442 | 53.892 |
| `poisson` | `Float32` | 10.328 | 59.007 | 0.471 | 75.243 |
| `bernoulli_logit` | `Float64` | 11.121 | 1.249 | 0.537 | 0.281 |
| `bernoulli_logit` | `Float32` | 9.559 | 1.808 | 0.597 | 0.186 |

<!-- BENCH:END -->

Heatmaps generated by CI (5x5 N x NPARAMS grid):

- `benchmarks/figures/fig1_forwarddiff.png`
- `benchmarks/figures/fig2_mooncake.png`
- `benchmarks/figures/fig3_enzyme.png`
- `benchmarks/figures/fig4_fastest_per_ppl.png`

## Development

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.test()
```

For benchmark sweeps:

```julia
julia --project=benchmarks benchmarks/sweep.jl
julia --project=benchmarks benchmarks/make_readme_table.jl
```
