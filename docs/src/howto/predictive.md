# Check a model with predictive draws

The examples use this model and a chain `chn` sampled from it:

```julia
@model function linreg(x, y)
    α ~ Normal(0, 10)
    β ~ Normal(0, 10)
    σ ~ Exponential(1)
    y .~ Normal.(α .+ β .* x, σ)
end
```

## Draw from the prior predictive

Pass missing data and call `rand`. Each draw holds the parameters and a
simulated `y`.

```julia
sims = rand(linreg(x, fill(missing, length(x))), 100)
sims[1].y
```

## Draw from the posterior predictive

Turn the chain into a vector of draws with [`chain_draws`](@ref), then call
[`predict`](@ref) on the model with missing data:

```julia
draws = chain_draws(chn)
preds = predict(linreg(x_new, fill(missing, length(x_new))), draws)
yrep = reduce(hcat, p.y for p in preds)    # one column per draw
```

For a `.~` statement the missing data must have the output's shape, as
`fill(missing, n)` does. A bare `missing` works for a `~` statement.

## Compute a derived quantity per draw

End the model with `return expr`, for example `return α + β` for the fitted
value at `x = 1`, then call [`returned`](@ref) on each draw:

```julia
fitted_at_1 = [returned(model, d) for d in draws]
```

## Get pointwise log-likelihoods for LOO

[`pointwise_loglikelihoods`](@ref) gives one log-likelihood per observation.
With `flatten=true`, stack them into the observations × draws matrix that
PSIS-LOO tools such as ParetoSmooth.jl take:

```julia
ll = reduce(hcat, pointwise_loglikelihoods(model, d; flatten=true) for d in draws)
```

For the summed value at one point, use [`loglikelihood_at`](@ref); for the
log joint, [`logjoint`](@ref).
