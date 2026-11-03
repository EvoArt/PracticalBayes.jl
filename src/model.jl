"""
    Model

A model with its arguments bound, as returned by calling a function defined
with [`@model`](@ref).

Fields:
- `f`: the evaluator the macro generated.
- `args`: the call's arguments as a `NamedTuple`. An argument holding
  `missing` is a parameter, not data.
- `conditioned`: data bound with [`condition`](@ref).
"""
struct Model{F,TArgs<:NamedTuple,TCond<:NamedTuple} <: AbstractPPL.AbstractProbabilisticProgram
    f::F           # the `_name_eval` function the @model macro generated (compiler.jl)
    args::TArgs
    conditioned::TCond
end

# Used by the @model-generated constructor (compiler.jl): `conditioned`
# starts empty and is only ever populated later via `condition`/`|`.
Model(f, args::NamedTuple) = Model(f, args, NamedTuple())

# `merge(m.conditioned, nt)` produces a NamedTuple whose TYPE encodes exactly
# which names are conditioned — that's the "type domain" mentioned above: two
# calls to `condition` with the same set of names (even different values)
# produce the same `TCond` type, so Julia only needs to specialize/compile
# the evaluator once per distinct *set of conditioned names*, not once per
# distinct value.
"""
    condition(model, nt::NamedTuple) -> Model
    model | nt -> Model

Return a copy of `model` in which the variables named in `nt` are observed with
the given values. Only whole variables can be conditioned, not elements.

```jldoctest
julia> @model function demo()
           μ ~ Normal(0, 1)
           y ~ Normal(μ, 1)
       end;

julia> m = demo() | (; y = 0.5);

julia> logjoint(m, (; μ = 0.0)) ≈ logpdf(Normal(), 0.0) + logpdf(Normal(), 0.5)
true
```
"""
condition(m::Model, nt::NamedTuple) = Model(m.f, m.args, merge(m.conditioned, nt))
Base.:|(m::Model, nt::NamedTuple) = condition(m, nt)

"""
    decondition(model, names::Symbol...) -> Model
    decondition(model) -> Model

Return a copy of `model` in which the variables `names`, previously bound with
[`condition`](@ref), are parameters again. With no names, all conditioning is
removed.
"""
decondition(m::Model) = Model(m.f, m.args, NamedTuple())
function decondition(m::Model, names::Symbol...)
    # `Base.structdiff` builds the NamedTuple of fields in `m.conditioned`
    # EXCLUDING `names` — i.e. the inverse operation of `merge`.
    kept = Base.structdiff(m.conditioned, NamedTuple{names})
    return Model(m.f, m.args, kept)
end

"""
    evaluate(model, mode::AbstractEvalMode, acc::Accum) -> (retval, acc)

Run the model body once under `mode`, starting from the accumulator `acc`.
Returns the body's return value and the final [`Accum`](@ref).

The higher-level functions ([`sample`](@ref), [`logjoint`](@ref),
[`predict`](@ref), ...) call this for you. See [Evaluation modes](@ref) for
what each mode does.
"""
@inline function evaluate(m::Model, mode::AbstractEvalMode, acc::Accum)
    # `m.f` is `_name_eval` from compiler.jl, whose fixed calling convention
    # is `(mode, acc, <the model's own positional args>...)`. `m.conditioned`
    # itself isn't passed here — the generated body reads conditioning off
    # `mode.conditioned` (via `getcond`), so it's the CALLER's responsibility
    # to construct `mode` with `mode.conditioned === m.conditioned` (done at
    # each mode-construction site: `build_layout`, `logdensity.jl`'s
    # `_logdensity_call`, etc) — `evaluate` itself doesn't enforce this.
    return _call_model(m.f, mode, acc, m.args)
end

# What PracticalBayes asks about a model after `@model` has expanded it: the
# closed-form gradient plan, the `:=` names, and whether `@static_model`
# defined it. `@model` answers with a one-argument method of the model's
# evaluator, `f(::_ModelInfo)` (see `_model_expr`, compiler.jl); anything else
# gets the defaults. Read at construction time and once per kept draw, never
# on the gradient path.
struct _ModelInfo end
const _NO_MODEL_INFO = (plan=nothing, walrus=(), static=false)
_model_info(f) = applicable(f, _ModelInfo()) ? f(_ModelInfo()) : _NO_MODEL_INFO

# TYPE ARGUMENTS. Turing models take element types as arguments,
# `@model function m(x, ::Type{T}=Vector{Float64}) where {T}`, and allocate with
# them (`z = T(undef, n); z[i] ~ ...`). DynamicPPL replaces a floating-point
# type argument with the AD number type on every evaluation, which is what lets
# such a container hold Duals; `paramtype(__mode__)` is PB's equivalent, and
# `_match_type` applies it to these arguments the same way.
#
# A type stored in a NamedTuple is typed as plain `DataType`, which would turn
# the evaluator call into a runtime dispatch. So the constructor stores it as
# a `TypeWrap{T}` (as DynamicPPL does), keeping `T` in the type domain, and
# `_call_model` unwraps it at compile time.
#
# `_call_model` writes the evaluator call out argument by argument (`f(mode,
# acc, getfield(args, 1), ...)`), so no argument tuple is built per call.
struct TypeWrap{T} end
_wrap_type(::Type{T}) where {T} = TypeWrap{T}()
_wrap_type(x) = x

@generated function _call_model(f, mode, acc, args::NamedTuple{names,V}) where {names,V}
    vals = map(eachindex(names)) do i
        t = V.parameters[i]
        t <: TypeWrap ? :(_match_type(mode, $(t.parameters[1]))) : :(getfield(args, $i))
    end
    return :(f(mode, acc, $(vals...)))
end

_match_type(mode, ::Type{T}) where {T} = T
_match_type(mode, ::Type{T}) where {T<:AbstractFloat} = paramtype(mode)
_match_type(mode, ::Type{Array{T,N}}) where {T<:AbstractFloat,N} = Array{paramtype(mode),N}
