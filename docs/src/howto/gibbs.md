# Sample with Gibbs blocks

A [`Gibbs`](@ref) sampler splits a model's parameters into blocks and updates
each block in turn, holding the others fixed. A block is updated by an HMC
sampler or by a Gibbs kernel, which draws the block's new values directly: a
built-in one such as [`ConjugateGibbs`](@ref) or [`PG`](@ref), or one you
write. Discrete parameters need a Gibbs kernel.

The example is a two-component mixture: each observation comes from
`Normal(μ, 1)` with probability `p` or from `Normal(0, 1)` otherwise, and the
allocations `z` are discrete.

```@example gibbs
using PracticalBayes, Distributions, StableRNGs, Statistics

y = vcat(randn(StableRNG(1), 60), 3 .+ randn(StableRNG(2), 40))

@model function mixture(y)
    p ~ Beta(1, 1)
    μ ~ Normal(0, 5)
    z = Vector{Bool}(undef, length(y))
    for i in eachindex(y)
        z[i] ~ Bernoulli(p)
    end
    y .~ Normal.(ifelse.(z, μ, 0.0), 1.0)
end
nothing # hide
```

## Write a Gibbs kernel

Subtype [`AbstractGibbsKernel`](@ref) and add a method to
[`gibbs_step`](@ref). `cond.values` holds the current value of every parameter,
and `cond.model.args` the model's arguments. Return a `NamedTuple` with the
block's new values.

```@example gibbs
struct AllocationKernel <: AbstractGibbsKernel end

function PracticalBayes.gibbs_step(rng, ::AllocationKernel, names, cond::ModelConditional)
    y, p, μ = cond.model.args.y, cond.values.p, cond.values.μ
    z = map(y) do yi
        l1 = log(p) + logpdf(Normal(μ, 1), yi)
        l0 = log1p(-p) + logpdf(Normal(0, 1), yi)
        rand(rng) < 1 / (1 + exp(l0 - l1))
    end
    return (; z = z)
end
nothing # hide
```

Return each value with the type it started with: here `Bool`, because
`rand(Bernoulli(p))` gives a `Bool`.

## Use a conjugate block

`p` has a Beta full conditional given `z`, so it can be drawn exactly with
[`ConjugateGibbs`](@ref). The function passed to it returns the counts of
successes and failures.

```@example gibbs
pk = ConjugateGibbs(:p, :beta, (1, 1)) do c
    s = count(c.values.z)
    (s, length(c.values.z) - s)
end
nothing # hide
```

## Use particle Gibbs or another sampler

For a discrete block with no exact kernel, [`PG`](@ref) draws it by particle
Gibbs: `:z => PG(50)`. A continuous block can also use any AbstractMCMC sampler
that works on a log density, such as AdvancedMH's random-walk sampler.

## Run the sampler

Give every parameter to exactly one block. To put several continuous
parameters in one HMC block, use a tuple of names: `(:a, :b) => NUTS(0.8)`.

```@example gibbs
spl = Gibbs(:μ => NUTS(0.8), :p => pk, :z => AllocationKernel())
chn = sample(StableRNG(3), mixture(y), spl, 1000;
             n_adapts=500, discard_initial=500, progress=false)
mean(chn[:μ]), mean(chn[:p])
```

The data were simulated with `μ = 3` and `p = 0.4`. Unlike a plain NUTS run, a
Gibbs run does no warm-up unless you ask: pass `n_adapts` and
`discard_initial`. `n_adapts` counts sweeps from the first one, so pass the
same value for the whole run.

## Keep large latent variables out of the chain

By default every sweep's `z` is stored in the chain. For a large latent
variable, use `save_states` to leave it out or write it to disk:

```julia
sample(model, spl, 1000; save_states=(z = :buffer,))         # not stored
using JLD2
sample(model, spl, 1000; save_states=(z = ("z.jld2", 100),)) # to disk every 100 sweeps
zs = read_states("z.jld2")
```

## Skip terms a block does not need

If an [`@addlogprob!`](@ref) term reads only some parameters, list them with
`depends`. An HMC block owning none of them then skips the term. Check the
lists with [`check_depends`](@ref):

```julia
@addlogprob! expensive_loglik(a, data) depends=(:a,)

check_depends(model, spl).ok
```

## Start from an earlier run's tuning

[`hmc_tuning`](@ref) reads the step size and metric each HMC block has reached
from a [`GibbsState`](@ref), and [`warm_start`](@ref) builds a sampler that
starts from them. The state comes from stepping the sampler yourself:

```julia
import AbstractMCMC, Random
rng = Random.default_rng()
spl = Gibbs(:μ => AdaptiveHMC(0.8; n_leapfrog=8), :p => pk, :z => AllocationKernel())

t, state = AbstractMCMC.step(rng, mixture(y), spl; n_adapts=500)
for _ in 2:1000
    t, state = AbstractMCMC.step(rng, mixture(y), spl, state; n_adapts=500)
end

spl2 = warm_start(spl, hmc_tuning(spl, state))
chn = sample(rng, mixture(y_new), spl2, 1000; init=state.values, n_adapts=100)
```

With `AdaptiveHMC` blocks, `n_adapts > 0` re-tunes only the step size.
`init=state.values` needs `y_new` to have the same length as `y`, since `z`
has one entry per observation.

## A larger example

[EpidemicTrajectories.jl](https://evoart.github.io/EpidemicTrajectories/dev/tutorials/cattle_iffbs/)
uses Gibbs kernels to sample the hidden infection states of an epidemic model
with forward filtering and backward sampling.
