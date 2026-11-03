# Port a model from Turing

Most Turing models run unchanged, and the
[Turing tutorials](https://turinglang.org/docs/tutorials/) can be followed with
PracticalBayes in place of Turing.

## Follow a Turing tutorial

Make three changes to the tutorial's code:

1. Replace `using Turing` with `using PracticalBayes, Distributions`.
   PracticalBayes does not re-export Distributions.
2. Replace `@varname(x)` with `:x`, since chains are indexed by symbol. An
   element such as `chn[@varname(m[1])]` becomes `param_draws(chn, :m, 1)`.
3. Replace `VNChain` with `SymChain` where a tutorial names the chain type.

With only these changes, these tutorials run: linear regression, logistic
regression, Poisson regression, multinomial logistic regression, time series
analysis, coin flipping, Bayesian neural networks, differential equations,
Gaussian processes (introduction) and probabilistic PCA.

Gaussian mixture models and hidden Markov models also need the indexed form of
change 2 for their plots; they sample with [`PG`](@ref). For a hidden Markov
model, an exact forward-filtering backward-sampling kernel is the better
choice; see [Sample with Gibbs blocks](@ref).

Infinite mixture models (`SMC`, `Turing.RandomMeasures`) and variational
inference (`Turing.Variational`) are not supported.

## Differences to know

### Prefer `.~` to a loop over the data

`y ~ dist`, `y .~ dist` and `y[i] ~ dist` observe `y` when `y` is a model
argument (or a `:=` name), and declare parameters otherwise. An argument the
body reassigns to a new container still counts as data. A loop of `y[i] ~`
works, but `.~` over the whole vector is much faster, and it accepts one
distribution per element where Turing needs `product_distribution`:

```julia
y .~ Normal.(mu, sigma)
```

`.~` cannot declare parameters; use an array distribution with `~`. For an IID
vector prior, `beta ~ MvNormal(...)` is much faster under reverse-mode AD than a
loop of `beta[j] ~`.

### Predictive sampling needs the output shape

For a `.~` or looped `y[i] ~` observe, pass `fill(missing, n)` for the data,
not a bare `missing`:

```julia
predict(model(X, fill(missing, n)), chn)
```

### `:=` also marks data

As in Turing, `z := expr` records `z` in the chain. It also makes `z` count as
data, so a later `z ~ dist` observes it. See [Data and parameters: `:=`](@ref).

### Only whole variables can be conditioned

`model | (; y = data)` works; `model | (; x[3] = 5.0)` does not.

### Float32 needs explicit literals

See [Run a model in Float32](@ref).

### Using both packages at once

PracticalBayes installs alongside current Turing, but many names, among them
`@model`, `NUTS`, `Gibbs`, `PG` and `filldist`, are exported by both. With both
loaded, `import` one and qualify its names:

```julia
using PracticalBayes
import Turing
```

### Vector parameters in chains

A vector parameter is one chain entry holding whole vectors, so
`chn[:beta][:, 2]` is the second chain, not `beta[2]`. Use
[`param_draws`](@ref):

```julia
param_draws(chn, :beta, 2)
```
