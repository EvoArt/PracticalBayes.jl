# Run a model in Float32

Three rules keep a model in Float32. Breaking one does not raise an error; the
computation is silently promoted to Float64.

- Write distribution constants as `Float32` literals: `Normal(0f0, 10f0)`,
  `Exponential(1.0f0)`.
- Create arrays from [`paramtype`](@ref): `zeros(paramtype(__mode__), k)`, not
  `zeros(k)`.
- Pass `NUTS(0.8f0)`, not `NUTS(0.8)`: the parameters take the sampler's
  number type, which comes from the target acceptance rate.

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

chn = sample(StableRNG(3), reg32(X, y), NUTS(0.8f0), 500; progress=false)
mean(chn[:beta])
```

The data were simulated with `beta = [1, -1, 0.5]`. Sampling runs in Float32,
but a bounded scalar such as `sigma` is reported in the chain as `Float64`.

To evaluate the log density and its gradient in Float32 yourself, build the
layout with `T=Float32`:

```julia
layout, θ0, store = build_layout(model; T=Float32)
ldf = LogDensityFunction(model, layout, store, AutoForwardDiff(); θ0=θ0)
```
