# PracticalBayes

[![CI](https://github.com/EvoArt/PracticalBayes.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/EvoArt/PracticalBayes.jl/actions/workflows/ci.yml)
[![Benchmark Sweep](https://github.com/EvoArt/PracticalBayes.jl/actions/workflows/benchmark.yml/badge.svg)](https://github.com/EvoArt/PracticalBayes.jl/actions/workflows/benchmark.yml)
[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://evoart.github.io/PracticalBayes.jl/)

PracticalBayes is a probabilistic programming package for Julia, and a work in
progress. It is built largely from the building blocks of the Turing ecosystem,
with its own model evaluation and some changes so that it works well with the
PracticallyBayesian packages. Models are written with Turing-style `@model` and
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

- **Gibbs kernels as plain functions.** A `Gibbs` block can be your own
  `AbstractGibbsKernel`: one `gibbs_step` method that receives the model's
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

See the [documentation](https://evoart.github.io/PracticalBayes.jl/) for a
getting-started guide, how-to guides and the API reference.

## Relationship to Turing

PracticalBayes is assembled from Turing ecosystem packages (AbstractPPL,
AdvancedHMC, Bijectors, Distributions, LogDensityProblems, FlexiChains). It has
its own model evaluation in place of DynamicPPL, does not depend on Turing
itself, and covers a narrower range of models. Most Turing models, and most of the
[Turing tutorials](https://turinglang.org/docs/tutorials/), run with small changes; see
[Port a model from Turing](https://evoart.github.io/PracticalBayes.jl/howto/from_turing/).

## Benchmarks

Benchmark results against Turing will be published here once they have been rerun.

<!-- BENCH:START -->
<!-- BENCH:END -->

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
