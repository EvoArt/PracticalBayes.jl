# A generic conjugate Gibbs kernel.
#
# Many Gibbs blocks have a closed-form full conditional in a conjugate family:
# given the rest of the state, the block's posterior is a Beta / Dirichlet / etc.
# whose parameters are the prior plus some sufficient statistics ("counts") read
# off the current draw of everything else. Writing such a kernel by hand is always
# the same shape — a struct, a `gibbs_step` method, a loop that accumulates the
# counts, and a draw — so this file provides the generic engine, leaving the user
# only the part that is actually model-specific: how to compute the counts.
#
# This is the conjugate sibling of the hand-written `FFBS` example in gibbs_kernel.jl:
# both are `AbstractGibbsKernel`s that `Gibbs` calls once per sweep, outside every
# gradient step, so they may use plain non-differentiated arithmetic throughout.
#
# It knows nothing about any particular model. The `count` closure receives the
# `ModelConditional` and reaches into `c.values` / `c.model.args` itself — so a
# domain package (e.g. an epidemic-model package) can build ready-made conjugate
# kernels on top of this by supplying count closures that know its data layout,
# without this engine assuming any of it.

using Distributions: Beta, Dirichlet

"""
    ConjugateGibbs(name, family, prior, count; n=nothing, collect=identity)

A Gibbs kernel that draws the block `name` from a conjugate Beta or Dirichlet
full conditional: the `prior` hyperparameters plus counts returned by `count`.

# Arguments
- `name`: the block's variable.
- `family`: `:beta` or `:dirichlet`.
- `prior`: `(a, b)` for `:beta`, a concentration vector for `:dirichlet`.
- `count`: `count(c)` returns the counts given the
  [`ModelConditional`](@ref) `c`: `(successes, failures)` for `:beta`, a vector
  of category counts for `:dirichlet`.
- `n`: for `n` independent draws (one per group, say), `count(c, k)` is called
  for each `k in 1:n`.
- `collect`: with `n`, turns the vector of draws into the block's value.

The prior in the model should match `prior`; the kernel does not read it.

```julia
pk = ConjugateGibbs(:p, :beta, (1, 1)) do c
    s = count(c.values.z)
    (s, length(c.values.z) - s)
end
spl = Gibbs(:p => pk, :μ => NUTS(0.8), :z => AllocationKernel())
```
"""
struct ConjugateGibbs{C,P,L} <: AbstractGibbsKernel
    name::Symbol
    family::Symbol
    prior::P
    count::C
    n::Union{Nothing,Int}
    collect::L
end
function ConjugateGibbs(name::Symbol, family::Symbol, prior, count;
                        n::Union{Nothing,Integer}=nothing, collect=identity)
    family in (:beta, :dirichlet) ||
        error("ConjugateGibbs: unknown conjugate family $(family) (want :beta or :dirichlet)")
    ConjugateGibbs(name, family, prior, count,
                   n === nothing ? nothing : Int(n), collect)
end

# Count-first method so the `do`-block form reads naturally — `do` passes the
# closure as the FIRST positional argument:
#     ConjugateGibbs(:p, :beta, (1, 1)) do c
#         ...counts...
#     end
ConjugateGibbs(count, name::Symbol, family::Symbol, prior; kwargs...) =
    ConjugateGibbs(name, family, prior, count; kwargs...)

# Draw one value of the given family from prior + counts. Dispatch on Val(family)
# keeps the hot path free of a runtime branch on the Symbol.
_conjugate_draw(rng, ::Val{:beta}, prior, counts) =
    rand(rng, Beta(prior[1] + counts[1], prior[2] + counts[2]))
_conjugate_draw(rng, ::Val{:dirichlet}, prior, counts) =
    rand(rng, Dirichlet(prior .+ counts))

function gibbs_step(rng, k::ConjugateGibbs, block_names, c::ModelConditional)
    block_names == (k.name,) ||
        error("ConjugateGibbs owns block $(k.name), got $(block_names)")
    fam = Val(k.family)
    if k.n === nothing
        draw = _conjugate_draw(rng, fam, k.prior, k.count(c))
        return NamedTuple{(k.name,)}((draw,))
    else
        draws = [_conjugate_draw(rng, fam, k.prior, k.count(c, i)) for i in 1:k.n]
        return NamedTuple{(k.name,)}((k.collect(draws),))
    end
end
