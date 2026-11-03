"""
    Model{F,TArgs<:NamedTuple,TCond<:NamedTuple} <: AbstractPPL.AbstractProbabilisticProgram

`f`: the generated evaluator, called as `f(mode, acc, args...)`.
`args`: the model-call arguments as a NamedTuple (values may be `missing` to
mark a parameter passed positionally, Turing-style).
`conditioned`: extra data bound via `model | (; y = ...)`, merged in on top of
`args` at each tilde site (see `getcond` in modes.jl).

The *role* of a top-level name (observed vs assumed) is determined by
whether it is a key of `args` (with a non-`missing`/non-`nothing` value) or
of `conditioned` — entirely in the type domain, so re-specializing on a new
conditioning pattern is ordinary Julia dispatch, not a runtime branch.
"""
struct Model{F,TArgs<:NamedTuple,TCond<:NamedTuple} <: AbstractPPL.AbstractProbabilisticProgram
    f::F           # the `_name_eval` function the @model macro generated (compiler.jl)
    args::TArgs
    conditioned::TCond
end

# Used by the @model-generated constructor (compiler.jl): `conditioned`
# starts empty and is only ever populated later via `condition`/`|`.
Model(f, args::NamedTuple) = Model(f, args, NamedTuple())

"""
    condition(model, nt::NamedTuple) -> Model
    model | nt -> Model

Returns a new `Model` with `nt` merged into `conditioned`. Re-specializes the
evaluator for the new conditioning pattern (a new `TCond` type), not per
value, so repeated conditioning with the same keys is cheap after the first
compile.
"""
# `merge(m.conditioned, nt)` produces a NamedTuple whose TYPE encodes exactly
# which names are conditioned — that's the "type domain" mentioned above: two
# calls to `condition` with the same set of names (even different values)
# produce the same `TCond` type, so Julia only needs to specialize/compile
# the evaluator once per distinct *set of conditioned names*, not once per
# distinct value.
condition(m::Model, nt::NamedTuple) = Model(m.f, m.args, merge(m.conditioned, nt))
Base.:|(m::Model, nt::NamedTuple) = condition(m, nt)

"""
    decondition(model, names::Symbol...) -> Model
    decondition(model) -> Model

Removes `names` from `conditioned`, reverting those sites to assumed. With no
names, removes ALL conditioning, as Turing's `decondition(model)` does — the
usual first step of `predict(decondition(model), chain)`.
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

Runs the model's evaluator under `mode`, threading `acc` through every tilde
site. This is the single entry point used by all four modes (`TraceMode`,
`EvalMode`, `PriorMode`, `FixedMode`).
"""
@inline function evaluate(m::Model, mode::AbstractEvalMode, acc::Accum)
    # `m.f` is `_name_eval` from compiler.jl, whose fixed calling convention
    # is `(mode, acc, <the model's own positional args>...)`. `m.conditioned`
    # itself isn't passed here — the generated body reads conditioning off
    # `mode.conditioned` (via `getcond`), so it's the CALLER's responsibility
    # to construct `mode` with `mode.conditioned === m.conditioned` (done at
    # each mode-construction site: `build_layout`, `logdensity.jl`'s
    # `_logdensity_call`, etc) — `evaluate` itself doesn't enforce this.
    return m.f(mode, acc, _eval_args(mode, m.args)...)
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
# `_eval_args` unwraps it at compile time. A model with no type arguments has
# no `TypeWrap` field, and `_eval_args` returns its arguments untouched.
struct TypeWrap{T} end
_wrap_type(::Type{T}) where {T} = TypeWrap{T}()
_wrap_type(x) = x

@generated function _eval_args(mode, args::NamedTuple{names,V}) where {names,V}
    any(t -> t <: TypeWrap, V.parameters) || return :args
    vals = map(eachindex(names)) do i
        t = V.parameters[i]
        t <: TypeWrap ? :(_match_type(mode, $(t.parameters[1]))) : :(getfield(args, $i))
    end
    return :(($(vals...),))
end

_match_type(mode, ::Type{T}) where {T} = T
_match_type(mode, ::Type{T}) where {T<:AbstractFloat} = paramtype(mode)
_match_type(mode, ::Type{Array{T,N}}) where {T<:AbstractFloat,N} = Array{paramtype(mode),N}
