# Reference

```@meta
CurrentModule = PracticalBayes
```

PracticalBayes also re-exports `HMC` and `HMCDA` from
[AdvancedHMC](https://github.com/TuringLang/AdvancedHMC.jl); `MCMCThreads`,
`MCMCSerial` and `MCMCDistributed` from AbstractMCMC; `AutoForwardDiff`,
`AutoReverseDiff`, `AutoMooncake` and `AutoEnzyme` from ADTypes; and `I` from
LinearAlgebra.

## Models

```@docs
@model
@static_model
@addlogprob!
Model
condition
decondition
```

## Distributions

```@docs
filldist
arraydist
Flat
FlatPos
LogPoisson
BinomialLogit
BernoulliCLogLog
```

## Sampling

```@docs
sample(::Random.AbstractRNG, ::Model, ::AdvancedHMC.AbstractHMCSampler, ::Integer)
sample(::Random.AbstractRNG, ::Model, ::Gibbs, ::Integer)
sample(::Random.AbstractRNG, ::Model, ::AbstractMCMC.AbstractSampler, ::Integer)
NUTS
SymChain
param_draws
chain_draws
NUTSthenHMC
AdaptiveHMC
JitteredNSteps
default_progress
setprogress!
```

## Gibbs sampling

```@docs
Gibbs
AbstractGibbsKernel
gibbs_step
ModelConditional
ConjugateGibbs
PG
GibbsState
AbstractMCMC.step(::Random.AbstractRNG, ::Model, ::Gibbs)
AbstractMCMC.step(::Random.AbstractRNG, ::Model, ::Gibbs, ::GibbsState)
hmc_tuning
warm_start
check_depends
DependsReport
SaveToChain
SaveToBuffer
SaveToDisk
read_states
write_state_chunk!
```

## Predictive draws and densities at a point

```@docs
Random.rand(::Random.AbstractRNG, ::Model)
Model(::Random.AbstractRNG)
predict
returned
logjoint
logprior
loglikelihood_at
pointwise_loglikelihoods
```

## Point estimates

```@docs
maximum_a_posteriori
maximum_likelihood
laplace_approximation
PointEstimate
LaplaceApproximation
laplace_mvnormal
```

## Log densities and gradients

```@docs
build_layout
Layout
link
invlink
LogDensityFunction
paramtype
AutoPBForwardDiff
GradMode
is_gradmode_compatible
check_gradmode
```

## Samplers for trimmed binaries

```@docs
static_nuts
static_hmc
static_gradient
StaticHMCResult
```

## Evaluation internals

```@docs
evaluate
Accum
loglikelihood_
AbstractEvalMode
TraceMode
EvalMode
PriorMode
FixedMode
PointwiseMode
ParticleMode
```
