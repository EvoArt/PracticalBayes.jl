module PracticalBayesPisteExt

# The gradient half of `AutoPBForwardDiff`, activated by `import Piste`.
#
# WHY THIS IS AN EXTENSION RATHER THAN A HARD DEPENDENCY
# ------------------------------------------------------
# `AutoPBForwardDiff` exists to make models trimmable, and `juliac --trim`
# requires Julia 1.12+. A user on 1.10 (still the LTS, and a version this
# package supports) can never trim anything, so they should not be made to
# install an AD engine they cannot use.
#
# Piste is registered in General (0.1.0); later releases may only be
# installable by URL until they are registered too. This method uses only
# `value_and_gradient!(g, f, x, Val(N))`, which is unchanged from 0.1 to 0.3.
#
# The backend TYPE stays in the package proper (`src/forwarddiff.jl`), so
# `AutoPBForwardDiff` can always be named and `chunksize` always works; only the
# method that actually differentiates lives here.

using PracticalBayes
using PracticalBayes: LogDensityFunction, Model, Layout, AutoPBForwardDiff,
                      chunksize, _PBFDObjective, _reject_value
using Piste: value_and_gradient!
import LogDensityProblems

# Same contract as every other backend in `logdensity.jl` — returns
# `(value, gradient)` with the same `reject_errors` semantics — so a sampler
# cannot tell which backend it is talking to.
function LogDensityProblems.logdensity_and_gradient(
    f::LogDensityFunction{M,L,S,<:AutoPBForwardDiff}, θ::AbstractVector
) where {M<:Model,L<:Layout,S<:NamedTuple}
    g = similar(θ)
    obj = _PBFDObjective(f.model, f.layout, f.store)
    # The chunk width is a TYPE parameter, recovered here as a `Val` so it stays
    # a compile-time constant all the way into Piste's inner loops. That is a
    # large part of why this trims: a runtime-valued chunk width is one of the
    # things that stops ForwardDiff from surviving the trimmer.
    chunk = Val(chunksize(f.adtype))
    if f.reject_errors
        try
            return value_and_gradient!(g, obj, θ, chunk)
        catch e
            e isa Union{InterruptException,OutOfMemoryError} && rethrow()
            # A zero gradient at `-Inf` is a safe "go nowhere" signal for any
            # sampler that checks the density before trusting the gradient.
            return _reject_value(θ), zeros(eltype(θ), length(θ))
        end
    end
    return value_and_gradient!(g, obj, θ, chunk)
end

end # module
