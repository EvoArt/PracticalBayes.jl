using AdvancedHMC:
    AdvancedHMC,
    AbstractHMCSampler,
    HMCKernel,
    Trajectory,
    EndPointTS,
    FixedNSteps,
    Leapfrog,
    Hamiltonian
using AdvancedHMC.Adaptation: StanHMCAdaptor, MassMatrixAdaptor, StepSizeAdaptor

"""
    AdaptiveHMC(δ=0.8; n_leapfrog=10, kwargs...)

Fixed-length HMC that adapts its step size and metric during the first
`n_adapts` iterations, as NUTS does, targeting acceptance rate `δ`. Every
iteration takes `n_leapfrog` leapfrog steps.

# Keywords
- `n_leapfrog`: leapfrog steps per iteration.
- `jitter`: draw the step count uniformly from `1:n_leapfrog` each iteration.
- `metric`: `:diagonal` (default), `:dense`, `:unit`, or an AdvancedHMC metric.
- `adapt`: `:full` adapts the metric and step size; `:step_size` keeps the
  metric and adapts only the step size (see [`warm_start`](@ref)).

```julia
chn = sample(model, AdaptiveHMC(0.8; n_leapfrog=16), 2000; n_adapts=1000)
```
"""
struct AdaptiveHMC{T<:Real,I,M} <: AbstractHMCSampler
    "Target acceptance rate for step-size dual averaging."
    δ::T
    "Fixed number of leapfrog steps per iteration, in both warm-up and sampling."
    n_leapfrog::Int
    "Whether to redraw the trajectory length uniformly from `1:n_leapfrog` each iteration."
    jitter::Bool
    "Choice of integrator (`Symbol` or `AbstractIntegrator`); `:leapfrog` searches for ϵ."
    integrator::I
    "Choice of initial metric (`Symbol` for auto-init, or an `AbstractMetric`)."
    metric::M
    "What warm-up adapts: `:full` (metric and step size) or `:step_size`."
    adapt::Symbol
end

function AdaptiveHMC(
    δ=0.8; n_leapfrog::Int=10, jitter::Bool=false, integrator=:leapfrog, metric=:diagonal,
    adapt::Symbol=:full,
)
    adapt in (:full, :step_size) ||
        throw(ArgumentError("adapt must be :full or :step_size, got :$adapt"))
    T = AdvancedHMC.determine_sampler_eltype(δ, integrator, metric)
    return AdaptiveHMC(T(δ), n_leapfrog, jitter, integrator, metric, adapt)
end

# Unlike `NUTSthenHMC`, this sampler needs no wrapper state and no `step` methods
# of its own: the kernel is a fixed-`n_leapfrog` leapfrog trajectory in BOTH
# phases, so AdvancedHMC's own generic `step` already does the right thing — it
# adapts while `i <= n_adapts` and freezes afterwards. We only have to fill in
# the four `make_*` hooks its initializer calls, so that (a) the kernel is
# fixed-step HMC rather than NUTS' tree sampler, and (b) the adaptor is the same
# `StanHMCAdaptor` NUTS uses, rather than `HMC`'s `NoAdaptation`.

AdvancedHMC.sampler_eltype(::AdaptiveHMC{T}) where {T} = T

# Metric and step-size handling are identical to any symbol-configured HMC
# sampler; AdvancedHMC's generic `make_metric`/`make_step_size` already dispatch
# on `spl.metric`/`spl.integrator`, so we lean on them unchanged. (No method
# needed here — the fallbacks in abstractmcmc.jl cover `AbstractHMCSampler`.)

# Fixed-length leapfrog trajectory, end-point sample — the same kernel
# `NUTSthenHMC` switches to after warm-up, here used from the first iteration.
function AdvancedHMC.make_kernel(spl::AdaptiveHMC, integrator::AdvancedHMC.AbstractIntegrator)
    return HMCKernel(
        Trajectory{EndPointTS}(integrator, _n_steps_criterion(spl.n_leapfrog, spl.jitter))
    )
end

# The piece plain `HMC` omits: a real Stan-style adaptor so warm-up learns the
# mass matrix (Welford) and the step size (dual averaging at target `δ`). The
# metric argument fixes what kind of mass matrix is estimated, so a `:dense`
# request adapts a full covariance, `:diagonal` a diagonal, `:unit` none.
function AdvancedHMC.make_adaptor(
    spl::AdaptiveHMC, metric::AdvancedHMC.AbstractMetric, integrator::AdvancedHMC.AbstractIntegrator
)
    spl.adapt === :step_size && return StepSizeAdaptor(spl.δ, integrator)
    return StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(spl.δ, integrator))
end
