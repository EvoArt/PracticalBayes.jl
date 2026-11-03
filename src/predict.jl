# M4 milestone: predictive utilities built on PriorMode/FixedMode (already
# fully implemented in modes.jl/tilde.jl — see those files' own docstrings
# for the exact per-site semantics each mode uses). This file is just the
# convenience layer that constructs the right mode and reads results back
# into user-facing NamedTuples/arrays.

using Random: Random

"""
    rand([rng,] model::Model) -> NamedTuple
    rand([rng,] model::Model, n::Integer) -> Vector{NamedTuple}

Draw from the prior predictive distribution. Parameters are drawn from their
priors; an observed variable keeps its data if it has any and is drawn from the
likelihood otherwise. Returns the parameters, the `:=` values and the drawn
observations.

To simulate a dataset, pass `missing` for the data argument, or
`fill(missing, n)` for a `.~` or looped `y[i] ~` statement.
"""
function Random.rand(rng::Random.AbstractRNG, m::Model)
    mode = PriorMode(rng, m.conditioned)
    evaluate(m, mode, Accum(0.0))
    return _narrow_missing(mode.values[])
end

# An indexed observe (`y[i] ~ dist` over a `missing`-filled argument) is
# recorded element by element into a `Union{Missing,T}` array (see
# `_record_element!` in tilde.jl), since nothing guarantees the loop visits
# every element. Once the model has run, an array with no `missing` left is
# narrowed to plain `T`, so a looped observe returns the same `Vector{Float64}`
# a `.~` observe does.
_narrow_missing(nt::NamedTuple) = map(_narrow_missing, nt)
function _narrow_missing(x::AbstractArray)
    (Missing <: eltype(x) && !any(ismissing, x)) || return x
    return convert(AbstractArray{nonmissingtype(eltype(x))}, x)
end
_narrow_missing(x) = x
Random.rand(m::Model) = rand(Random.default_rng(), m)
Random.rand(rng::Random.AbstractRNG, m::Model, n::Integer) = [rand(rng, m) for _ in 1:n]
Random.rand(m::Model, n::Integer) = rand(Random.default_rng(), m, n)

# Shared by `returned`/`predict`/`logjoint`/`logprior`/`loglikelihood`: builds
# a `FixedMode` at a given NamedTuple of parameter values and runs the model.
# `predict` gates whether un-conditioned observe sites are sampled (true) or
# must already have real data (false, the default — used by the log-density
# accessors, where a missing observation genuinely can't be scored).
function _fixed_evaluate(rng, m::Model, fixed::NamedTuple; predict::Bool=false)
    mode = FixedMode(rng, fixed, m.conditioned; predict=predict)
    retval, acc = evaluate(m, mode, Accum(0.0))
    return retval, acc, _narrow_missing(mode.values[])
end

"""
    logjoint(model::Model, nt::NamedTuple) -> Real

The log joint density of `model` with its parameters set to `nt`, for example
one posterior draw. Every observed variable must have data.

```jldoctest
julia> @model function demo(y)
           μ ~ Normal(0, 1)
           y ~ Normal(μ, 1)
       end;

julia> logjoint(demo(0.5), (; μ = 0.0)) ≈ logpdf(Normal(), 0.0) + logpdf(Normal(), 0.5)
true
```
"""
function logjoint(m::Model, nt::NamedTuple; rng=Random.default_rng())
    _, acc, _ = _fixed_evaluate(rng, m, nt; predict=false)
    return logjoint(acc)
end
"""
    logprior(model::Model, nt::NamedTuple) -> Real

The log prior density of `model` with its parameters set to `nt`.
"""
function logprior(m::Model, nt::NamedTuple; rng=Random.default_rng())
    _, acc, _ = _fixed_evaluate(rng, m, nt; predict=false)
    return logprior(acc)
end
"""
    loglikelihood_at(model::Model, nt::NamedTuple) -> Real

The log-likelihood of `model`'s data with its parameters set to `nt`. Every
observed variable must have data. For one value per observation, use
[`pointwise_loglikelihoods`](@ref).
"""
function loglikelihood_at(m::Model, nt::NamedTuple; rng=Random.default_rng())
    _, acc, _ = _fixed_evaluate(rng, m, nt; predict=false)
    return loglikelihood_(acc)
end

"""
    returned(model::Model, nt::NamedTuple; rng=Random.default_rng())

Run `model` with its parameters set to `nt` and return the value of the model
body's `return` statement (`nothing` if it has none). Use it to compute derived
quantities at each posterior draw.

```jldoctest
julia> @model function demo()
           m̃ ~ Normal(0, 1)
           return m̃ + 1
       end;

julia> returned(demo(), (; m̃ = 2.0))
3.0
```
"""
function returned(m::Model, nt::NamedTuple; rng=Random.default_rng())
    retval, _, _ = _fixed_evaluate(rng, m, nt; predict=false)
    return retval
end

"""
    predict([rng,] model::Model, draws) -> Vector{NamedTuple}

Posterior predictive sampling. For each `NamedTuple` in `draws` (see
[`chain_draws`](@ref)), set the parameters to that draw and draw every observed
variable that has no data from the likelihood.

Pass `model` with the data argument replaced by missing values. For a `.~` or
looped `y[i] ~` statement give the output shape, `fill(missing, n)`; a bare
`missing` works only for `y ~`. An array with only some elements missing is not
supported.

Returns one `NamedTuple` per draw holding every variable.

```julia
chn = sample(linreg(x, y), NUTS(0.8), 1000)
preds = predict(linreg(x_new, fill(missing, length(x_new))), chain_draws(chn))
```
"""
function predict(rng::Random.AbstractRNG, m::Model, draws)
    return [_fixed_evaluate(rng, m, nt; predict=true)[3] for nt in draws]
end
predict(m::Model, draws) = predict(Random.default_rng(), m, draws)

"""
    predict([rng,] model::Model, chn::SymChain; include_all=false) -> SymChain

Posterior predictive sampling from a chain. For every draw in `chn`, set the
parameters to that draw and draw each observed variable that has no data.
Returns a chain of the predicted variables, with the same iterations and chains
as `chn`; `include_all=true` adds the parameters.

Free the observations by passing missing data,
`predict(linreg(x_new, fill(missing, n)), chn)`, or by removing conditioning,
`predict(decondition(model), chn)`.
"""
function predict(rng::Random.AbstractRNG, m::Model, chn::FlexiChains.SymChain; include_all::Bool=false)
    params = _fixed_params(m, chn)
    niters, nchains = size(chn[first(params)])
    vals = [_fixed_evaluate(rng, m, _draw_at(chn, params, i, c); predict=true)[3]
            for i in 1:niters, c in 1:nchains]
    # The predicted variables are the ones the model DREW or COMPUTED: not a
    # parameter held fixed, and not an observe site that already had data.
    # `:=` values are recomputed rather than held, so `yhat := X * beta` follows
    # the predictive model's new `X`.
    names = [k for k in keys(vals[1]) if (include_all || !(k in params)) && !_has_data(m, k)]
    data = Dict(FlexiChains.Parameter(k) => [getfield(v, k) for v in vals] for k in names)
    return FlexiChains.FlexiChain{Symbol}(niters, nchains, data;
                                         iter_indices=collect(Int, FlexiChains.iter_indices(chn)),
                                         chain_indices=collect(Int, FlexiChains.chain_indices(chn)))
end
predict(m::Model, chn::FlexiChains.SymChain; kwargs...) = predict(Random.default_rng(), m, chn; kwargs...)

"""
    returned(model::Model, chn::SymChain) -> Matrix

The model body's return value at every draw of `chn`, as an
`(iterations × chains)` matrix.
"""
function returned(m::Model, chn::FlexiChains.SymChain; rng=Random.default_rng())
    params = _fixed_params(m, chn)
    niters, nchains = size(chn[first(params)])
    return [returned(m, _draw_at(chn, params, i, c); rng=rng) for i in 1:niters, c in 1:nchains]
end

# The chain parameters to hold fixed: everything except the model's `:=`
# names, which a chain stores per draw but which the model computes itself.
_fixed_params(m::Model, chn) = filter(p -> !(p in walrus_names(m.f)), FlexiChains.parameters(chn))

# One draw of `chn` as the NamedTuple of parameter values a FixedMode reads.
_draw_at(chn, params, i, c) = NamedTuple{Tuple(params)}(Tuple(chn[p][i, c] for p in params))

# Whether site `k` is observed data: conditioned, or a model argument that
# holds a value (not `missing`/`nothing`, nor an array that is all `missing`,
# which is how a caller asks for that argument to be predicted).
function _has_data(m::Model, k::Symbol)
    hasproperty(m.conditioned, k) && return true
    hasproperty(m.args, k) || return false
    v = getproperty(m.args, k)
    return !(v === missing || v === nothing || (v isa AbstractArray && any(ismissing, v)))
end

"""
    (model::Model)([rng])

Run `model` once with every parameter drawn from its prior and return the
model body's return value, as calling a Turing model does (`model()`).
Observe sites with data keep their data. For the drawn values themselves, use
`rand(model)`.
"""
function (m::Model)(rng::Random.AbstractRNG=Random.default_rng())
    retval, _ = evaluate(m, PriorMode(rng, m.conditioned), Accum(0.0))
    return retval
end

"""
    chain_draws(chn::FlexiChains.SymChain) -> Vector{NamedTuple}

Every draw in `chn` as a `NamedTuple` of parameter values, all chains pooled
(chain by chain). Sampler statistics are left out. The result is the `draws`
argument [`predict`](@ref) expects, and each element can be passed to
[`logjoint`](@ref), [`returned`](@ref) or [`pointwise_loglikelihoods`](@ref).
"""
function chain_draws(chn::FlexiChains.SymChain)
    param_names = FlexiChains.parameters(chn)
    niters, nchains = size(chn[first(param_names)])
    draws = Vector{NamedTuple}(undef, niters * nchains)
    idx = 1
    for c in 1:nchains, i in 1:niters
        draws[idx] = NamedTuple{Tuple(param_names)}(Tuple(chn[p][i, c] for p in param_names))
        idx += 1
    end
    return draws
end

"""
    param_draws(chn::FlexiChains.SymChain, name::Symbol) -> Matrix
    param_draws(chn::FlexiChains.SymChain, name::Symbol, i) -> Matrix

The draws of one parameter as a plain `(draws × chains)` `Matrix`. For a
vector-valued parameter, pass the element index `i`.

A vector parameter is one chain entry whose values are whole vectors, so
`chn[:beta][:, 2]` indexes the second chain, not `beta[2]`. Use this function
instead:

```julia
param_draws(chn, :beta, 2)     # every draw of beta[2], all chains
param_draws(chn, :sigma)       # a scalar parameter
```
"""
function param_draws(chn::FlexiChains.SymChain, name::Symbol)
    b = chn[name]
    eltype(b) <: AbstractArray && throw(ArgumentError(
        "`$(name)` is vector-valued (eltype $(eltype(b))); pass an element " *
        "index, e.g. `param_draws(chn, :$(name), 1)`"))
    # Plain `1:size(...)` ranges rather than `axes(b, ...)` on purpose: a
    # comprehension over a DimArray's own axes comes back as a DimArray, and
    # this is documented to return a plain Matrix.
    return [b[i, c] for i in 1:size(b, 1), c in 1:size(b, 2)]
end

function param_draws(chn::FlexiChains.SymChain, name::Symbol, i)
    b = chn[name]
    eltype(b) <: AbstractArray || throw(ArgumentError(
        "`$(name)` is scalar-valued; drop the index and call " *
        "`param_draws(chn, :$(name))`"))
    # Every draw has the same length (the layout is fixed), so checking the
    # first is enough to give a clear error instead of a BoundsError deep in
    # the comprehension.
    n = length(first(b))
    (i isa Integer && 1 <= i <= n) || throw(BoundsError(first(b), i))
    # plain ranges, not `axes` -- see the scalar method above
    return [b[j, c][i] for j in 1:size(b, 1), c in 1:size(b, 2)]
end

"""
    pointwise_loglikelihoods(model::Model, nt::NamedTuple; flatten=false)
        -> NamedTuple or Vector{Float64}

The log-likelihood of each observation with the parameters set to `nt`, as
needed for PSIS-LOO or WAIC. Every observed variable must have data.

Returns a `NamedTuple` with one vector per observed variable, or, with
`flatten=true`, one vector of all values in the order the model visits them.

```jldoctest
julia> @model function demo(y)
           μ ~ Normal(0, 1)
           y .~ Normal(μ, 1)
       end;

julia> pointwise_loglikelihoods(demo([0.0, 1.0]), (; μ = 0.0)).y ≈ logpdf.(Normal(), [0.0, 1.0])
true
```
"""
function pointwise_loglikelihoods(m::Model, nt::NamedTuple; flatten::Bool=false)
    mode = PointwiseMode(nt, m.conditioned)
    evaluate(m, mode, Accum(0.0))
    pw = mode.pointwise[]
    flatten || return pw
    return reduce(vcat, values(pw))
end
