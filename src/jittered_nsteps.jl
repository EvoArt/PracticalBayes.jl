using AdvancedHMC:
    AdvancedHMC,
    Trajectory,
    FixedNSteps,
    StaticTerminationCriterion,
    PhasePoint,
    Hamiltonian
using Random: Random, AbstractRNG

"""
    JitteredNSteps(L)

AdvancedHMC termination criterion for a fixed-length trajectory whose step
count is drawn uniformly from `1:L` at each iteration. A drop-in replacement for
`FixedNSteps(L)` in a `Trajectory{EndPointTS}`. It avoids a fixed trajectory
length resonating with the posterior, and costs `(L + 1) / 2` steps per
iteration on average.

[`AdaptiveHMC`](@ref) and [`NUTSthenHMC`](@ref) use it when `jitter=true`.
"""
mutable struct JitteredNSteps <: StaticTerminationCriterion
    "Maximum number of leapfrog steps; each iteration draws uniformly from `1:L`."
    L::Int
    "The step count drawn for the current iteration; read by `nsteps`."
    current::Int
end

function JitteredNSteps(L::Int)
    L >= 1 || throw(ArgumentError("L must be at least 1, got $L"))
    # Seed `current` with `L` so that a criterion which somehow reaches `nsteps`
    # before any `transition` has drawn for it still yields a valid trajectory
    # (the fixed-`L` one) rather than a zero-step no-op.
    return JitteredNSteps(L, L)
end

# Both of AdvancedHMC's `nsteps(τ)` call sites land here and read the value the
# `transition` below drew for this iteration.
AdvancedHMC.nsteps(τ::Trajectory{TS,I,TC}) where {TS,I,TC<:JitteredNSteps} =
    τ.termination_criterion.current

# The single point where the per-iteration draw happens. We draw, then swap in a
# plain `FixedNSteps` holding that draw and hand straight over to AdvancedHMC's
# own static-trajectory `transition`: accept/reject, momentum reversal, energy
# bookkeeping and stats assembly are all unchanged and stay in AdvancedHMC where
# they belong.
#
# Rebuilding the `Trajectory` around a `FixedNSteps` (rather than `invoke`-ing the
# generic method on our own type) keeps the dispatch honest -- the delegated call
# is an ordinary call to an existing public method, with no dependence on the
# internal signature of the method we would otherwise be bypassing.
#
# This method is strictly more specific than AdvancedHMC's
# (`TC<:StaticTerminationCriterion`), so it takes precedence for `JitteredNSteps`
# without touching the `FixedNSteps` path at all.
function AdvancedHMC.transition(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    h::Hamiltonian,
    tau::Trajectory{TS,I,TC},
    z::PhasePoint,
) where {TS<:AdvancedHMC.AbstractTrajectorySampler,I,TC<:JitteredNSteps}
    tc = tau.termination_criterion
    # `rand(rng, 1:L)` on a vector of RNGs (AdvancedHMC's matrix-parallel mode)
    # would be ambiguous about which RNG to use, so take the first one there;
    # all chains in that mode already share a coupled trajectory length.
    r = rng isa AbstractVector ? first(rng) : rng
    L = rand(r, 1:tc.L)
    # Keep `current` in step with the draw so `nsteps(tau)` on THIS object still
    # reports the trajectory that was simulated, for anything that inspects the
    # criterion directly rather than going through the delegated trajectory.
    tc.current = L
    fixed = Trajectory{TS}(tau.integrator, FixedNSteps(L))
    return AdvancedHMC.transition(rng, h, fixed, z)
end

"""
    _n_steps_criterion(L, jitter)

Pick the termination criterion for a static HMC trajectory of nominal length `L`:
`JitteredNSteps` when `jitter` is set, AdvancedHMC's plain `FixedNSteps`
otherwise. Shared by `AdaptiveHMC` and `NUTSthenHMC` so the two samplers cannot
drift apart in what `jitter=true` means.
"""
_n_steps_criterion(L::Int, jitter::Bool) =
    jitter ? JitteredNSteps(L) : FixedNSteps(L)
