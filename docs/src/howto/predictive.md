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

Call [`predict`](@ref) on the model with missing data. It returns a chain of
the predicted variables:

```julia
preds = predict(linreg(x_new, fill(missing, length(x_new))), chn)
yrep = reduce(hcat, preds[:y])    # one column per draw
```

For a `.~` or looped `y[i] ~` statement the missing data must have the
output's shape, as `fill(missing, n)` does. A bare `missing` works for `y ~`.
To free observations bound with [`condition`](@ref), use
`predict(decondition(model), chn)`.

## Compute a derived quantity per draw

End the model with `return expr`, for example `return α + β` for the fitted
value at `x = 1`. [`returned`](@ref) gives its value at every draw, as a
draws × chains matrix:

```julia
fitted_at_1 = returned(model, chn)
```

## Get pointwise log-likelihoods for LOO

[`pointwise_loglikelihoods`](@ref) gives one log-likelihood per observation at
one draw; [`chain_draws`](@ref) turns the chain into a vector of draws.
With `flatten=true`, stack them into the observations × draws matrix that
PSIS-LOO tools such as ParetoSmooth.jl take:

```julia
ll = reduce(hcat, pointwise_loglikelihoods(model, d; flatten=true) for d in chain_draws(chn))
```

For the summed value at one point, use [`loglikelihood_at`](@ref); for the
log joint, [`logjoint`](@ref).
