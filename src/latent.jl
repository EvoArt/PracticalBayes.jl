# Latent-kernel contract for Gibbs (gibbs.jl): a user plugs in a custom
# sampler for a block of variables that can't (or shouldn't) go through
# HMC/NUTS — a discrete latent state, a whole trajectory sampled via exact
# forward-filtering/backward-sampling (FFBS), etc. Deliberately minimal: no
# VarName/context machinery, just plain NamedTuple field access, so writing
# a kernel is "one struct, one method" (see `latent_step`'s docstring for a
# worked example).

"""
    AbstractLatentKernel

Supertype for samplers you write yourself for a block of a [`Gibbs`](@ref)
sampler. A subtype must implement [`latent_step`](@ref).

[`Gibbs`](@ref) calls the kernel once per sweep and never differentiates
through it, so a kernel may use any code: discrete draws, exact
forward-filtering backward-sampling, Metropolis proposals.

Mutable state inside a kernel must survive `deepcopy`, which `MCMCThreads`
uses to give each chain its own sampler.
"""
abstract type AbstractLatentKernel end

"""
    ModelConditional(model, values)

The state a [`latent_step`](@ref) call conditions on.

- `c.values`: the current value of every parameter, including the block's own.
- `c.model.args`: the model's arguments, which hold the data.
- `c.model.conditioned`: data bound with [`condition`](@ref).
"""
struct ModelConditional{M<:Model,V<:NamedTuple}
    model::M
    values::V
end

"""
    logjoint(c::ModelConditional; overrides...) -> Real

The model's log joint density at the current values, with any parameters given
as keywords replaced. A Metropolis kernel can use it to score a proposal:

```julia
lp_new = logjoint(c; z = z_proposed)
```

It runs the whole model, so an exact update is usually faster.
"""
function logjoint(c::ModelConditional; overrides...)
    nt = merge(c.values, NamedTuple(overrides))
    mode = FixedMode(Random.default_rng(), nt, c.model.conditioned; predict=false)
    _, acc = evaluate(c.model, mode, Accum(0.0))
    return logjoint(acc)
end

"""
    latent_step(rng, kernel::AbstractLatentKernel, block_names::Tuple{Vararg{Symbol}},
                c::ModelConditional) -> NamedTuple

Draw new values for the block `block_names` given the rest of the state in
`c` (a [`ModelConditional`](@ref)). Implement this for your
[`AbstractLatentKernel`](@ref) subtype.

Return a `NamedTuple` with exactly the keys in `block_names`. Each value must
have the same type as the variable's starting value, e.g. `Bool` for a
`Bernoulli` variable and `Int` for a `Categorical` one.

The call happens once per Gibbs sweep, outside any gradient computation, so the
values in `c` are plain numbers.

# Example

A two-component mixture whose allocations `z` are drawn exactly given `p` and
`μ`:

```julia
@model function mixture(y)
    p ~ Beta(1, 1)
    μ ~ Normal(0, 5)
    z = Vector{Bool}(undef, length(y))
    for i in eachindex(y)
        z[i] ~ Bernoulli(p)
    end
    y .~ Normal.(ifelse.(z, μ, 0.0), 1.0)
end

struct AllocationKernel <: AbstractLatentKernel end

function PracticalBayes.latent_step(rng, ::AllocationKernel, block_names, c::ModelConditional)
    y, p, μ = c.model.args.y, c.values.p, c.values.μ
    z = map(y) do yi
        l1 = log(p) + logpdf(Normal(μ, 1), yi)
        l0 = log1p(-p) + logpdf(Normal(0, 1), yi)
        rand(rng) < 1 / (1 + exp(l0 - l1))
    end
    return (; z = z)
end

spl = Gibbs(:p => NUTS(0.8), :μ => NUTS(0.8), :z => AllocationKernel())
chn = sample(mixture(y), spl, 1000; n_adapts=500, discard_initial=500)
```
"""
function latent_step end
