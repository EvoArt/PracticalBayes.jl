# Port a model from Turing

Most Turing models run unchanged. These are the differences.

## Which statements observe data

As in Turing, `y ~ dist`, `y .~ dist` and `y[i] ~ dist` observe `y` when `y`
is a model argument (or a `:=` name), and declare parameters otherwise. An
argument the body reassigns to a new container still counts as data.

A loop of `y[i] ~` works, but `.~` over the whole vector is much faster:

```julia
y .~ Normal.(mu, sigma)            # faster than: for i in eachindex(y); y[i] ~ Normal(mu[i], sigma); end
```

## `.~` takes per-element distributions, and only observes

`y .~ Normal.(eta, sigma)` with a vector `eta` works, where Turing needs
`product_distribution`. `.~` cannot declare parameters; use an array
distribution with `~`. For an IID vector prior, `beta ~ MvNormal(...)` is much
faster under reverse-mode AD than a loop of `beta[j] ~`.

## Predictive sampling needs the output shape

For a `.~` or looped `y[i] ~` observe, pass `fill(missing, n)` for the data,
not a bare `missing`:

```julia
predict(model(X, fill(missing, n)), chain_draws(chn))
```

## Warm-up draws are kept unless you discard them

Turing drops the adaptation draws by default; PracticalBayes keeps them. Pass
`discard_initial`:

```julia
sample(model, NUTS(0.8), 1000; n_adapts=1000, discard_initial=1000)
```

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
