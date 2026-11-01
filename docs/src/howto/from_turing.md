# Port a model from Turing

Most Turing models run unchanged. These are the differences.

## Observe data with `y ~` or `y .~`, not `y[i] ~`

An indexed statement `x[i] ~ dist` always declares a parameter, even when `x`
is a data argument. Observe data as a whole:

```julia
y .~ Normal.(mu, sigma)            # instead of: for i in eachindex(y); y[i] ~ ...; end
y ~ MvNormal(X * beta, sigma^2 * I)
```

## `.~` takes per-element distributions, and only observes

`y .~ Normal.(eta, sigma)` with a vector `eta` works, where Turing needs
`product_distribution`. `.~` cannot declare parameters; use an array
distribution with `~`. For an IID vector prior, `beta ~ MvNormal(...)` is much
faster under reverse-mode AD than a loop of `beta[j] ~`.

## Predictive sampling at a `.~` statement needs the output shape

Pass `fill(missing, n)` for the data, not a bare `missing`:

```julia
predict(model(X, fill(missing, n)), chain_draws(chn))
```

## Warm-up draws stay in the chain

`sample(model, NUTS(0.8), N; n_adapts)` returns all `N` iterations. Drop the
warm-up with `chn[iter=(n_adapts + 1):N]`. A `Gibbs` run honours
`discard_initial`.

## Only whole variables can be conditioned

`model | (; y = data)` works; `model | (; x[3] = 5.0)` does not.

## Float32 needs explicit literals

See [Run a model in Float32](@ref).

## Model arguments cannot be `Val` type parameters

`f(::Val{x}) where {x}` is not supported. Pass a plain value and branch on it.

## Names that clash

`@model`, `Model`, `LogDensityFunction`, `filldist` and `arraydist` are
exported by both packages. With both loaded, `import` one and qualify its
names:

```julia
using PracticalBayes
import Turing
```

## Chains

[`sample`](@ref) returns a FlexiChains [`SymChain`](@ref), as Turing 0.45 does.
A vector parameter is one entry holding whole vectors, so `chn[:beta][:, 2]` is
the second chain, not `beta[2]`. Use [`param_draws`](@ref):

```julia
param_draws(chn, :beta, 2)
```

Multiple chains work as in Turing, with `MCMCThreads` from AbstractMCMC:

```julia
using AbstractMCMC: MCMCThreads
chn = sample(model, NUTS(0.8), MCMCThreads(), N, 4)
```
