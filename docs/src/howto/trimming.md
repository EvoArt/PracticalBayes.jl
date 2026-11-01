# Compile a model into a standalone binary

Julia 1.12's `juliac --trim` can compile a simple PracticalBayes model and its
sampler into an executable that runs without Julia installed. The repository's
working example is `trimtest/PBTrim`: two parameters, sampled with
[`static_hmc`](@ref).

## What a trimmed program can use

- [`@static_model`](@ref) for the model.
- A layout built at precompile time and stored in a `const`. The binary cannot
  call [`build_layout`](@ref).
- A [`LogDensityFunction`](@ref) built without an AD backend, for the density.
- A gradient you supply yourself.
- [`static_nuts`](@ref) or [`static_hmc`](@ref) as the sampler.

AdvancedHMC's samplers, DifferentiationInterface backends such as ForwardDiff,
and [`GradMode`](@ref) do not trim. [`AutoPBForwardDiff`](@ref) is written to
trim, but the working example does not use it.

## Write the program

```julia
module MyTrim

using PracticalBayes
using Distributions: Normal, Exponential
using LogDensityProblems: LogDensityProblems
using Random: Xoshiro

@static_model function simple(y)
    mu ~ Normal(0, 1)
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end

const M = simple(2.0)
const LAYOUT, THETA0, STORE0 = build_layout(M; rng=Xoshiro(1234))
const LDF = LogDensityFunction(M, LAYOUT, STORE0; θ0=THETA0)

logp(x::Vector{Float64}) = LogDensityProblems.logdensity(LDF, x)

# Central differences keep the example short; an analytic gradient is faster.
function logp_grad!(g::Vector{Float64}, x::Vector{Float64})
    xp = copy(x)
    for i in eachindex(x)
        h = 1e-6 * max(1.0, abs(x[i]))
        xp[i] = x[i] + h; fp = logp(xp)
        xp[i] = x[i] - h; fm = logp(xp)
        xp[i] = x[i]
        g[i] = (fp - fm) / (2h)
    end
    return logp(x)
end

function (@main)(args::Vector{String})::Cint
    d = length(THETA0)
    res = static_hmc(logp_grad!, copy(THETA0), ones(d), 0.05, 16, 500, 200; seed=1)
    println(Core.stdout, "mean of mu: ", sum(res.draws[1, :]) / size(res.draws, 2))
    return 0
end

end
```

Seed the RNG passed to `build_layout`: the starting point is drawn from it and
baked into the binary, so an unseeded RNG gives a different binary on each
build. Tune `eps`, `L` and `inv_mass` for [`static_hmc`](@ref) beforehand, for
example with [`static_nuts`](@ref) or [`static_gradient`](@ref) under the JIT.

## Build and run

With [JuliaC.jl](https://github.com/JuliaLang/JuliaC.jl) installed in an
environment `juliac-env`:

```
julia --project=juliac-env -e 'using JuliaC; JuliaC.main(ARGS)' -- \
    --output-exe mytrim --bundle build --trim=safe --experimental ./MyTrim
JULIA_LOAD_CODEGEN_LIB=0 ./build/bin/mytrim
```

Zero verifier errors does not guarantee the binary works. Run it with
`JULIA_LOAD_CODEGEN_LIB=0` and compare its output with the same code under
the JIT.
