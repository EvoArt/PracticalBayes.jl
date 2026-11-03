# PracticalBayes.jl

PracticalBayes is a WIP probabilistic programming package for Julia. It is built largely from the building blocks of the Turing ecosystem
(AbstractPPL, AdvancedHMC, Bijectors, Distributions, LogDensityProblems and
FlexiChains), with its own model evaluation and some changes so that it works
well with the PracticallyBayesian packages.

Models are written with Turing-style `@model` and `~` syntax and sampled with
HMC/NUTS, alone or as blocks of a Gibbs sampler.

```julia
using PracticalBayes, Distributions

@model function linreg(x, y)
    α ~ Normal(0, 10)
    β ~ Normal(0, 10)
    σ ~ Exponential(1)
    y .~ Normal.(α .+ β .* x, σ)
end

chn = sample(linreg(x, y), NUTS(), 1000)
```

## What is different here

- **Gibbs kernels as plain functions.** A `Gibbs` block can be your own
  [`AbstractGibbsKernel`](@ref): one `gibbs_step` method that receives the
  model's data and the current value of every parameter, and returns new values
  for its block. It runs once per MCMC step, outside the gradient. It is similar
  to Turing's `GibbsConditional` and `externalsampler`, but designed to be easy
  to use with the latent-state samplers of epidemic models, such as those in
  [EpidemicTrajectories.jl](https://evoart.github.io/EpidemicTrajectories/dev/tutorials/cattle_iffbs/). See [Sample with Gibbs blocks](@ref).
- **Float32.** A model can run in Float32 end to end, with a few rules about
  literals. See [Run a model in Float32](@ref).
- **Trimmable models.** A simple model can be compiled into a standalone
  binary with `juliac --trim`, with some restrictions. See
  [Compile a model into a standalone binary](@ref).
- **GPU data (untested).** Data can be held on a CUDA device. This has not
  yet been run on GPU hardware and is a work in progress. See
  [Keep data on a GPU](@ref).

## Where to go next

- New to the package: [Getting started](@ref).
- Coming from Turing: [Port a model from Turing](@ref). The
  [Turing tutorials](https://turinglang.org/docs/tutorials/) can be followed
  with PracticalBayes with minimal changes.
- How the pieces fit together: the [Explanation](@ref).
