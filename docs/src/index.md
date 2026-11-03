# PracticalBayes.jl

PracticalBayes is a probabilistic programming package for Julia, and a work in
progress. It is built largely from the building blocks of the Turing ecosystem
(AbstractPPL, AdvancedHMC, Bijectors, Distributions, LogDensityProblems and
FlexiChains), with its own model evaluation and some changes so that it works
well with the PracticallyBayesian packages. It does not use DynamicPPL or
Turing itself.

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

chn = sample(linreg(x, y), NUTS(0.8), 1000; n_adapts=1000, discard_initial=1000)
```

Install it from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/EvoArt/PracticalBayes")
```

## What is different here

- **Samplers you write yourself, outside the gradient.** A block of a `Gibbs`
  sampler can be your own [`AbstractLatentKernel`](@ref), alongside NUTS and
  conjugate blocks. It runs once per sweep and is never differentiated, so it
  can update discrete or very large latent variables with any code. See
  [Sample with Gibbs blocks](@ref).
- **Float32.** A model can run in Float32 end to end, with a few rules about
  literals. See [Run a model in Float32](@ref).
- **Trimmable models.** A simple model can be compiled into a standalone
  binary with `juliac --trim`, with some restrictions. See
  [Compile a model into a standalone binary](@ref).
- **GPU data (untested).** Data can be held on a CUDA device. This has not
  yet been run on GPU hardware and is a work in progress. See
  [Keep data on a GPU](@ref).

## Where to go next

- New to the package: the [Tutorial](@ref).
- Coming from Turing: [Port a model from Turing](@ref).
- Every exported name: the [Reference](@ref).
- How the pieces fit together: the [Explanation](@ref).
