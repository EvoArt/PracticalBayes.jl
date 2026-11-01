# Find the MAP estimate or a Laplace approximation

Point estimates need Optimization.jl and a solver package:

```julia
using PracticalBayes, Optimization, OptimizationOptimJL

est = maximum_a_posteriori(model, LBFGS())
est.constrained          # estimates by name
est.value                # log joint at the estimate
```

[`maximum_likelihood`](@ref) takes the same arguments and ignores the priors.

To hold some parameters fixed, pass their values as `store`:

```julia
est = maximum_a_posteriori(model, LBFGS(); store=(; σ = 0.3))
```

[`laplace_approximation`](@ref) adds a normal approximation around the mode.
It is on the unconstrained scale, so map draws back with [`invlink`](@ref):

```julia
la = laplace_approximation(model, LBFGS())
d = laplace_mvnormal(la)
draws = [invlink(la.map.layout, θ) for θ in eachcol(rand(d, 1000))]
```
