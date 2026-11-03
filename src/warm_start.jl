using AdvancedHMC: DenseEuclideanMetric, DiagEuclideanMetric, UnitEuclideanMetric

"""
    hmc_tuning(spl::Gibbs, state::GibbsState) -> Tuple

The metric and step size each HMC block of `spl` has reached in `state`, one
entry per block: `nothing` for a latent block, otherwise
`(names, metric, step_size)`. `metric` is the inverse mass matrix as a plain
array (a `Matrix` for a dense metric, a `Vector` for a diagonal one, `nothing`
for the unit metric).

Plain arrays rather than AdvancedHMC objects, so the tuning serialises and can
start a related run through [`warm_start`](@ref): the same model refitted to
slightly different data needs little or no warm-up once it starts from a metric
learnt on the first.
"""
function hmc_tuning(spl::Gibbs, state::GibbsState)
    length(state.subs) == length(spl.blocks) || throw(ArgumentError(
        "state has $(length(state.subs)) blocks but the sampler has $(length(spl.blocks))"))
    return ntuple(length(spl.blocks)) do i
        sub = state.subs[i]
        sub isa GibbsSamplerSub || return nothing
        hs = sub.hmc_state
        hs isa AdvancedHMC.HMCState || throw(ArgumentError(
            "block $i ($(join(spl.blocks[i].names, ", "))) has no AdvancedHMC state to read " *
            "a tuning from; hmc_tuning needs a state that has taken at least one sweep"))
        (names = spl.blocks[i].names, metric = _metric_array(hs.metric),
         step_size = AdvancedHMC.nom_step_size(hs.κ.τ.integrator))
    end
end

_metric_array(m::DenseEuclideanMetric) = Matrix(m.M⁻¹)
_metric_array(m::DiagEuclideanMetric) = Vector(m.M⁻¹)
_metric_array(::UnitEuclideanMetric) = nothing

_metric_object(a::AbstractMatrix) = DenseEuclideanMetric(Matrix{Float64}(a))
_metric_object(a::AbstractVector) = DiagEuclideanMetric(Vector{Float64}(a))

"""
    warm_start(spl::Gibbs, tuning; adapt=:step_size) -> Gibbs

`spl` with each HMC block starting from the metric and step size in `tuning`
(from [`hmc_tuning`](@ref)) instead of a unit metric and a step-size search.
Pair it with `init = state.values` to start from where the earlier run ended as
well.

What warm-up then does is set by `n_adapts` at the `sample` call:

- `n_adapts = 0`: the carried metric and step size are used as they are.
- `n_adapts > 0` and `adapt = :step_size`: the step size is re-tuned by dual
  averaging and the metric is kept. This is the cheap choice when the target
  has moved a little, since the step size is what a shift in scale upsets.
- `n_adapts > 0` and `adapt = :full`: the whole windowed adaptation runs again.
  The windowed estimate of the metric restarts from its first window, so this
  buys little over a cold start beyond skipping the step-size search.

`:step_size` needs [`AdaptiveHMC`](@ref) blocks. A `NUTS` block takes the
carried metric and step size under `:full` only. A plain `HMC` block adapts
nothing, so it is left as it is.
"""
function warm_start(spl::Gibbs, tuning; adapt::Symbol=:step_size)
    adapt in (:full, :step_size) ||
        throw(ArgumentError("adapt must be :full or :step_size, got :$adapt"))
    length(tuning) == length(spl.blocks) || throw(ArgumentError(
        "tuning has $(length(tuning)) blocks but the sampler has $(length(spl.blocks))"))
    blocks = ntuple(length(spl.blocks)) do i
        b, tu = spl.blocks[i], tuning[i]
        tu === nothing && return b
        _is_latent_block(b) && throw(ArgumentError(
            "tuning has an entry for block $i, which is a latent block"))
        Tuple(tu.names) == b.names || throw(ArgumentError(
            "tuning for block $i is for ($(join(tu.names, ", "))), " *
            "but the block holds ($(join(b.names, ", ")))"))
        GibbsBlock(b.names, _warm_kernel(b.kernel, tu, adapt))
    end
    return Gibbs(blocks)
end

function _warm_kernel(k::AdaptiveHMC, tu, adapt::Symbol)
    metric = tu.metric === nothing ? k.metric : _metric_object(tu.metric)
    integrator = Leapfrog(Float64(tu.step_size))
    T = AdvancedHMC.determine_sampler_eltype(k.δ, integrator, metric)
    return AdaptiveHMC(T(k.δ), k.n_leapfrog, k.jitter, integrator, metric, adapt)
end

function _warm_kernel(k::AdvancedHMC.NUTS, tu, adapt::Symbol)
    adapt === :full || throw(ArgumentError(
        "adapt = :step_size needs AdaptiveHMC blocks; NUTS always runs its full " *
        "adaptation, so pass adapt = :full"))
    metric = tu.metric === nothing ? k.metric : _metric_object(tu.metric)
    return AdvancedHMC.NUTS(k.δ, k.max_depth, k.Δ_max, Leapfrog(Float64(tu.step_size)), metric)
end

# Plain HMC adapts nothing: its metric and step size are the ones it was built
# with, so there is nothing to carry.
_warm_kernel(k::AdvancedHMC.HMC, tu, adapt::Symbol) = k
# `NUTS(; adtype=...)`: warm the sampler inside, keep the block's backend.
_warm_kernel(k::_ADSampler, tu, adapt::Symbol) = _ADSampler(_warm_kernel(k.sampler, tu, adapt), k.adtype)

_warm_kernel(k, tu, adapt::Symbol) = throw(ArgumentError(
    "warm_start supports AdaptiveHMC, NUTS and HMC blocks, not $(nameof(typeof(k)))"))
