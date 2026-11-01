# Tutorial

In this tutorial we fit a straight line to simulated data: we write a model,
sample its posterior with NUTS, look at the result, and check the fit with
posterior predictive draws.

## Simulate some data

We need PracticalBayes and Distributions. StableRNGs makes the random numbers
the same on every machine.

```@example tutorial
using PracticalBayes, Distributions, StableRNGs, Statistics

rng = StableRNG(1)
x = collect(range(0, 1; length=50))
y = 1.0 .+ 2.0 .* x .+ 0.3 .* randn(rng, 50)
nothing # hide
```

The true intercept is 1, the slope 2 and the noise standard deviation 0.3.

## Write the model

A model is a function marked with `@model`. Each `~` line gives a variable a
distribution. `α`, `β` and `σ` are not arguments, so they are parameters. `y`
is an argument holding data, so `.~` observes each element of it.

```@example tutorial
@model function linreg(x, y)
    α ~ Normal(0, 10)
    β ~ Normal(0, 10)
    σ ~ Exponential(1)
    y .~ Normal.(α .+ β .* x, σ)
end

model = linreg(x, y)
nothing # hide
```

## Sample

We run NUTS for 2000 iterations. The first 1000 adapt the step size and are
then dropped.

```@example tutorial
chn = sample(StableRNG(2), model, NUTS(0.8), 2000; n_adapts=1000, progress=false)
chn = chn[iter=1001:2000]
nothing # hide
```

## Look at the result

The chain is indexed by parameter name. Each entry is a matrix with one row per
draw and one column per chain.

```@example tutorial
mean(chn[:α]), mean(chn[:β]), mean(chn[:σ])
```

The posterior means are close to the values we simulated from.

## Check the fit

To draw new data from the fitted model we build it again with the data replaced
by missing values. `fill(missing, 50)` gives the shape of the output.
[`predict`](@ref) then draws `y` once for each posterior draw.

```@example tutorial
draws = chain_draws(chn)
preds = predict(StableRNG(3), linreg(x, fill(missing, 50)), draws)
yrep = reduce(hcat, p.y for p in preds)
size(yrep)
```

Each column of `yrep` is one simulated dataset. If the model fits, about 90% of
the observations should fall inside their 90% predictive intervals:

```@example tutorial
lo = [quantile(r, 0.05) for r in eachrow(yrep)]
hi = [quantile(r, 0.95) for r in eachrow(yrep)]
mean(lo .<= y .<= hi)
```

## Next steps

- Mix NUTS with your own samplers: [Sample with Gibbs blocks](@ref).
- More predictive checks and LOO: [Check a model with predictive draws](@ref).
- Every function used here: the [Reference](@ref).
