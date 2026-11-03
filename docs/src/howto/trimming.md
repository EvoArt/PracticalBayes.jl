# Compile a model into a standalone binary

Julia 1.12's `juliac --trim` can compile a simple PracticalBayes model, its
gradient and its sampler into an executable that runs without Julia installed.
The gradient comes from Piste, PracticalBayes' trimmable AD backend.

This is experimental. It has been shown to work on small models like the one
below; check your own model's binary against the same code run normally.

## Write the program

Put the program in its own package, with PracticalBayes, Piste, Distributions,
LogDensityProblems and Random as dependencies:

```julia
module MyTrim

using PracticalBayes
import Piste
using Distributions: Normal, Exponential
using LogDensityProblems: LogDensityProblems
using Random: Xoshiro

@static_model function simple(y)
    mu ~ Normal(0, 1)
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end

# A trimmed binary cannot inspect the model when it runs, so the log density
# is built once, when the package is precompiled.
const M = simple(2.0)
const LAYOUT, THETA0, STORE0 = build_layout(M; rng=Xoshiro(1234))
const LDF = LogDensityFunction(M, LAYOUT, STORE0, AutoPBForwardDiff(chunk=2); θ0=THETA0)

function logp_grad!(g::Vector{Float64}, x::Vector{Float64})
    lp, grad = LogDensityProblems.logdensity_and_gradient(LDF, x)
    copyto!(g, grad)
    return lp
end

function (@main)(args::Vector{String})::Cint
    res = static_hmc(logp_grad!, copy(THETA0), ones(length(THETA0)), 0.05, 16, 500, 200; seed=1)
    println(Core.stdout, "mean of mu: ", sum(res.draws[1, :]) / size(res.draws, 2))
    return 0
end

end
```

Points to keep:

- Define the model with [`@static_model`](@ref), not `@model`.
- Use [`AutoPBForwardDiff`](@ref) for the gradient. Other AD backends, such as
  ForwardDiff, and the closed-form [`GradMode`](@ref) gradient do not trim.
- Sample with [`static_hmc`](@ref). [`sample`](@ref) and AdvancedHMC's samplers
  do not trim.
- Seed the RNG for the starting point, so every build starts from the same
  place.
- Choose the step size, step count and metric for [`static_hmc`](@ref)
  beforehand: run the program under Julia, with
  `MyTrim.main(String[])`, and check `res.n_divergent` and `res.accept_rate`.

## Build and run

Install [JuliaC.jl](https://github.com/JuliaLang/JuliaC.jl) in an environment
`juliac-env`, then:

```
julia --project=juliac-env -e 'using JuliaC; JuliaC.main(ARGS)' -- \
    --output-exe mytrim --bundle build --trim=safe --experimental ./MyTrim
JULIA_LOAD_CODEGEN_LIB=0 ./build/bin/mytrim
```

A build with no verifier errors can still be wrong. Run the binary with
`JULIA_LOAD_CODEGEN_LIB=0` and check that its output matches
`MyTrim.main(String[])` run under Julia.
