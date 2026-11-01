# Run a model in Float32

## Write the model for Float32

Three rules keep a model in Float32. Breaking one does not raise an error; the
computation is silently promoted to Float64.

- Write distribution constants as `Float32` literals: `Normal(0f0, 10f0)`,
  `Exponential(1.0f0)`.
- Create arrays from [`paramtype`](@ref): `zeros(paramtype(__mode__), k)`, not
  `zeros(k)`.
- Pass `NUTS(0.8f0)`, not `NUTS(0.8)`: the sampler takes its number type from
  the target acceptance rate.

Data can stay `Float64`.

```@example f32
using PracticalBayes, Distributions, LinearAlgebra, StableRNGs, Statistics

@model function reg32(X, y)
    k = size(X, 2)
    beta ~ MvNormal(zeros(paramtype(__mode__), k), 1f0 * I)
    sigma ~ Exponential(1f0)
    y ~ MvNormal(X * beta, sigma^2 * I)
end

X = randn(StableRNG(1), 100, 3)
y = X * [1.0, -1.0, 0.5] .+ 0.2 .* randn(StableRNG(2), 100)
model = reg32(X, y)
nothing # hide
```

## Sample

[`sample`](@ref) builds `Float64` parameters, so build the Float32 log density
yourself and sample it through AbstractMCMC:

```@example f32
using ADTypes, LogDensityProblems
import AbstractMCMC

layout, θ0, store = build_layout(model; T=Float32)
ldf = LogDensityFunction(model, layout, store, AutoForwardDiff(); θ0=θ0)
LogDensityProblems.logdensity_and_gradient(ldf, θ0)
```

```@example f32
ts = AbstractMCMC.sample(StableRNG(3), AbstractMCMC.LogDensityModel(ldf), NUTS(0.8f0), 1000;
                         n_adapts=500, initial_params=θ0, progress=false)
draws = [invlink(layout, t.z.θ) for t in ts[501:end]]
mean(d.beta for d in draws)
```

The data were simulated with `beta = [1, -1, 0.5]`. The draws are
`NamedTuple`s of `Float32` values, which [`predict`](@ref) and
[`logjoint`](@ref) accept directly.
