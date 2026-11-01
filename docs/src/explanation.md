# Explanation

## Latent kernels run outside the gradient

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
