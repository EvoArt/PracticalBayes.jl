# Explanation

## Latent kernels in a Gibbs sampler

HMC moves continuous parameters using the gradient of the log density, and each
iteration evaluates that gradient many times. A discrete or very large latent
variable, such as a hidden state trajectory, has no gradient, and
differentiating through it would be expensive even if it did.

A [`Gibbs`](@ref) sampler therefore splits the work. Each sweep updates the
blocks in turn. An HMC block sees every other block as fixed data: those values
sit in a store passed to the AD backend as constants, so they are never
differentiated, and an HMC step never calls a latent kernel. A latent kernel
runs once per sweep with plain numbers, so it can use any algorithm: exact
conditional draws, forward filtering and backward sampling, or Metropolis
proposals scored with [`logjoint`](@ref).

The cost is the usual one for Gibbs sampling: blocks that are strongly
correlated mix slowly when updated separately.

## Data and parameters: `:=`

Whether `x ~ dist` observes data or declares a parameter is decided when
`@model` expands the body, from names alone: `x` is data if it is one of the
model's arguments, and a parameter otherwise. Deciding this once, rather than
checking at every evaluation, is part of what keeps an observe statement as
cheap as a `logpdf` call.

The rule needs a way to say that a name computed inside the body is data too.
That is what `:=` does. `z := expr` runs as an ordinary assignment, and also adds
`z` to the names treated as data for the rest of the body, so a later
`z ~ dist` (or `z[i] ~ dist`) observes the computed value:

```julia
@model function f(x)
    μ ~ Normal(0, 1)
    z := x .- 1      # data, computed from an argument
    z .~ Normal(μ, 1)
end
```

With `=` instead, `z ~ dist` would declare `z` a parameter. A `:=` value is not
stored in the chain; use [`returned`](@ref) for derived quantities.

## Skipping terms in Gibbs blocks: `depends=`

An HMC block of a [`Gibbs`](@ref) sampler evaluates the model's whole log
density and its gradient with respect to the block's own parameters. A term
that does not involve any of those parameters is a constant for that block: it
shifts the log density by a fixed amount and adds nothing to the gradient, so
evaluating it is wasted work. For an expensive [`@addlogprob!`](@ref) term,
such as the likelihood of a latent trajectory, that waste can dominate a block
that owns only a few cheap parameters.

`@addlogprob! expr depends=(:a, :b)` says which variables `expr` reads. Within a
block, the check is resolved when the code is compiled: if the block owns none
of `a` and `b`, the term is removed from that block's evaluation entirely.
Skipping it is exact, since the term could not change that block's update.

The annotation is not checked automatically. If it leaves out a variable the
term reads, the blocks owning that variable lose a real part of their gradient
and sample the wrong distribution, without any error. [`check_depends`](@ref)
compares each block's gradient with and without skipping and reports any
difference. A whole-model sampler such as plain `NUTS` owns every variable, so
it never skips anything.

## Evaluation modes

`@model` turns the model body into a function that takes an evaluation mode as
its first argument. Every `~` statement asks the mode what to do: where the
parameter's value comes from and what to add to the log density. The mode is a
type, so for a given model and mode Julia compiles the choice away and an
observe statement costs what `logpdf` costs.

| Mode | Used by | Parameters come from |
|---|---|---|
| [`TraceMode`](@ref) | [`build_layout`](@ref) | `init` or the prior |
| [`EvalMode`](@ref) | [`LogDensityFunction`](@ref), sampling | the vector `θ` or the store |
| [`PriorMode`](@ref) | `rand(model)` | the prior |
| [`FixedMode`](@ref) | [`logjoint`](@ref), [`returned`](@ref), [`predict`](@ref) | a `NamedTuple` |
| [`PointwiseMode`](@ref) | [`pointwise_loglikelihoods`](@ref) | a `NamedTuple` |

`PriorMode`, and `FixedMode` when called from `predict`, also draw observed
variables that have no data.

[`build_layout`](@ref) runs the model once in `TraceMode` to find its
parameters and fix where each lives: a range of the unconstrained vector `θ`,
or the store of constants. Sampling then only ever runs `EvalMode` with that
fixed layout.

## Gradient paths

[`LogDensityFunction`](@ref) gets its gradient in one of three ways.

- **AD through DifferentiationInterface.** Any backend DifferentiationInterface
  supports, chosen with `adtype`: ForwardDiff by default, or Mooncake, Enzyme
  and others. The gradient is prepared once, for the element type of `θ0`.
- **The closed form.** When `@model` sees that a model is a generalised linear
  model, with a linear predictor that is visibly linear in the parameters, it
  records the model's structure. The gradient is then computed directly,
  essentially `X' * residual`, without AD. This happens automatically; the
  recognition is purely syntactic and conservative, so a model it does not
  recognise uses AD. See [`GradMode`](@ref) and [`check_gradmode`](@ref).
- **Piste forward mode.** [`AutoPBForwardDiff`](@ref) computes derivatives with
  dual numbers whose sizes are fixed at compile time and needs nothing built at
  run time, which is what `juliac --trim` requires.

The static samplers, [`static_nuts`](@ref) and [`static_hmc`](@ref), exist for
the same reason: AdvancedHMC does not compile under `--trim`, so these are
written with no dependencies and take a gradient function you supply. See
[Compile a model into a standalone binary](@ref).
