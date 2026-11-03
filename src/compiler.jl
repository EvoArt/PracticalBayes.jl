using MacroTools: MacroTools, @capture, postwalk, splitdef, combinedef

"""
    @model function name(args...; kwargs...) ... end

Define a model. Calling `name(args...)` returns a [`Model`](@ref).

Statements in the body:

- `x ~ dist`: if `x` is a model argument holding data, `x` is observed and
  `logpdf(dist, x)` is added to the log-likelihood. Otherwise `x` is a
  parameter. Pass `missing` for an argument to make it a parameter, or bind
  data later with [`condition`](@ref).
- `x[i] ~ dist`: if `x` is a model argument, the element `x[i]` is observed.
  Otherwise `x[i]` is one element of an indexed family of parameters; create
  the container first, for continuous elements with
  `x = Vector{paramtype(__mode__)}(undef, n)` (see [`paramtype`](@ref)).
- `y .~ dists`: observe every element of the argument `y` at once, faster than
  a loop. `dists` may be one distribution or a broadcast such as
  `Normal.(mu, sigma)`. `.~` cannot introduce parameters; use an array
  distribution with `~` for those.
- `z := expr`: assign `z`, record it in the chain, and treat it as data, so a
  later `z ~ dist` observes it.
- [`@addlogprob!`](@ref)` expr`: add `expr` to the log-likelihood.
- `return expr`: the value [`returned`](@ref) gives back.

Discrete parameters (`Categorical`, `Bernoulli`, ...) cannot be sampled by HMC;
give them a Gibbs kernel in [`Gibbs`](@ref).

```julia
@model function linreg(x, y)
    α ~ Normal(0, 10)
    β ~ Normal(0, 10)
    σ ~ Exponential(1)
    y .~ Normal.(α .+ β .* x, σ)
end

m = linreg(x, y)
chn = sample(m, NUTS(0.8), 1000)
```
"""
macro model(expr)
    return _model_expr(expr)
end

# The expansion behind both `@model` and `@static_model`; they differ only in
# `static`, the default layout form `build_layout` uses for the model.
function _model_expr(expr; static::Bool=false)
    # `splitdef` (MacroTools) parses `function name(args...; kwargs...) body end`
    # into a Dict with keys :name, :args, :kwargs, :whereparams, :body, ... —
    # much less fragile than hand-rolling AST pattern matching for every
    # variant of function-definition syntax (short form, where-clauses, etc).
    def = splitdef(expr)
    name = def[:name]
    # Anonymous typed arguments (Turing's `::Type{T}=Float64` idiom) get a name,
    # so the evaluator and `Model.args` can carry them like any other argument.
    def[:args] = _name_anonymous_args(def[:args])
    # `:=`'d names are folded into the same `argnames` set used for the
    # data-vs-parameter decision — from the tilde-rewriter's point of view,
    # "declared as a model argument" and "already computed via `z := expr`
    # earlier in the body" are the same thing: a name that's guaranteed to
    # already hold a concrete value by the time it's referenced.
    argnames = union(_argnames(def), _walrus_names(def[:body]))

    eval_name = Symbol("_", name, "_eval")
    # Passes, in order: (1) turn `z := expr` into plain `z = expr` — by the
    # time this runs `argnames` has already recorded `z`, so the special
    # syntax has done its job and reduces to an ordinary assignment; (2) turn
    # every `~`/`.~` into a call into tilde.jl; (3) make every `return expr`
    # also return the threaded accumulator. `_rewrite_returns` must run last
    # so it also catches any `return` the user wrote *around* a tilde
    # statement (rare, but legal Julia).
    body = _rewrite_walrus(def[:body])
    body = _rewrite_tildes(body, argnames)
    body = _rewrite_addlogprob(body)
    body = _rewrite_returns(body)

    # Build the evaluator's `Dict` in the same shape `splitdef` produces, so
    # `combinedef` can turn it back into a function-definition Expr. We prepend
    # `__mode__` and `__acc__` as the first two positional arguments — every
    # generated evaluator has this fixed calling convention, checked by
    # `evaluate(model, mode, acc)` in model.jl (`m.f(mode, acc, m.args...)`).
    #
    # `AbstractEvalMode`/`Accum` are referenced fully module-qualified
    # (`PracticalBayes.AbstractEvalMode`) rather than escaped: the whole
    # generated block gets wrapped in one `esc(quote ... end)` at the bottom
    # (so user identifiers like `Normal` resolve at the macro call site), and
    # `esc` is not composable — escaping a piece that's already inside a
    # fully-escaped quote produces invalid syntax ("escape (escape ...)").
    # Module-qualifying is the standard way to reach into our own package's
    # internals from inside an escaped quote without re-escaping.
    #
    # The qualifier is the module OBJECT, interpolated (`$(PracticalBayes).x`),
    # never the bare name `PracticalBayes`: escaped code resolves names in the
    # CALLER's module, and a caller that wrote `import PracticalBayes as PB`
    # has no binding called `PracticalBayes` there, so every generated model
    # failed with `UndefVarError`. The same applies to every `$(PracticalBayes)`
    # in this file.
    # `Model.args` is always ONE flat NamedTuple, splatted POSITIONALLY into
    # the evaluator by `evaluate` (`m.f(mode, acc, m.args...)`, model.jl) — so
    # the evaluator itself takes every model argument positionally, keyword
    # args included, in a fixed order (original positional args, then
    # original keyword args by their declared order). Only the user-facing
    # CONSTRUCTOR keeps the real `function name(...; kw...)` signature (so
    # `coinflip(; N=5)` keeps working as call syntax); it repackages whatever
    # it received into that same fixed positional order before handing off to
    # `Model`. Bare `def[:kwargs]` on the evaluator would NOT work here: a
    # positional splat of a NamedTuple can't bind to a `function f(; k)`
    # keyword parameter.
    eval_arg_names = vcat(map(_argsym, def[:args]), map(_argsym, def[:kwargs]))
    # Arguments are bare names on the evaluator, EXCEPT where the annotation
    # mentions a `where` parameter: that annotation is the only thing that can
    # bind the parameter (`::Type{T}` binds `T`, `::Val{k}` binds `k`), and the
    # body may use it.
    wnames = _where_names(get(def, :whereparams, ()))
    eval_pos_args = map(def[:args]) do a
        ann = _annotation(a)
        ann !== nothing && _mentions(ann, wnames) ? :($(_argsym(a))::$(ann)) : _argsym(a)
    end
    eval_def = Dict(
        :name => eval_name,
        :args => [
            :(__mode__::$(PracticalBayes).AbstractEvalMode), :(__acc__::$(PracticalBayes).Accum),
            eval_pos_args..., map(_argsym, def[:kwargs])...,
        ],
        :kwargs => [],
        :whereparams => def[:whereparams],
        :body => quote
            $(body)
            return (nothing, __acc__)  # reached only if the user's body falls through without an explicit `return`
        end,
    )
    eval_fn = combinedef(eval_def)

    # The user-facing constructor keeps the original argument list (no
    # __mode__/__acc__) and just packages the call args into a NamedTuple for
    # `Model.args`, in the SAME (positional-args-then-kwargs) order the
    # evaluator above expects. Missing/positional data vs parameters is NOT
    # decided here — it's decided per tilde-site by `_tilde_expansion` below,
    # using `argnames` (static, macro-time) plus `getcond` (runtime, but folds
    # away for a concrete conditioning-pattern type).
    ctor_args = def[:args]
    ctor_kwargs = def[:kwargs]
    # `_wrap_type` (model.jl) keeps a type-valued argument in the type domain;
    # it is the identity on everything else.
    nt_pairs = [:($(a) = $(PracticalBayes)._wrap_type($(a))) for a in eval_arg_names]

    ctor_def = Dict(
        :name => name,
        :args => ctor_args,
        :kwargs => ctor_kwargs,
        :whereparams => def[:whereparams],
        :body => quote
            return $(PracticalBayes).Model($(eval_name), (; $(nt_pairs...)))
        end,
    )
    ctor_fn = combinedef(ctor_def)

    # `esc(...)` wraps the *entire* generated code so it is spliced into the
    # macro call site's environment rather than into PracticalBayes' own
    # module scope — necessary so that e.g. `Normal` in the user's model body
    # resolves via the user's `using Distributions`, not ours.
    # GradMode (gradmode_recognize.jl): try to recognize this model body as a
    # GLM with a closed-form gradient. This runs at MACRO-EXPANSION time on the
    # ORIGINAL body (before tilde rewriting), so it costs nothing at run time
    # and nothing on the evaluation path.
    #
    # Recognition failing is the normal, expected case and is NOT an error —
    # `gradmode_plan` then returns `nothing` and everything falls back to
    # general AD.
    plan = try
        recognize_glm(def[:body], argnames)
    catch
        # A recognizer bug must never break `@model` itself. Swallowing here
        # means the worst case is "no fast path", never "your model won't
        # compile".
        nothing
    end
    # What PracticalBayes later asks about the model — the closed-form plan
    # (`gradmode_plan`), the `:=` names (`walrus_names`), the default layout
    # form (`prefers_static_layout`) — is answered by a one-argument method of
    # the EVALUATOR itself (`_model_info`, model.jl), not by methods added to
    # those PracticalBayes functions. `@model` is used inside function bodies
    # (a model built by a constructor function), and defining a method of
    # another module's function there is a syntax error; a method of the
    # evaluator, which is local there too, is not. It also leaves `Model`'s
    # fields and every construction site as they were.
    info = (plan=plan, walrus=_walrus_list(def[:body]), static=static)
    info_fn = :($(eval_name)(::$(PracticalBayes)._ModelInfo) = $(QuoteNode(info)))

    return esc(quote
        $(eval_fn)
        $(ctor_fn)
        $(info_fn)
    end)
end

# Pulls the bare argument name out of either `x` or `x::T` argument syntax.
# `splitdef`'s :args entries are Exprs like `:(x::Int)` for typed positional
# args; `:kwargs` entries are additionally either `Expr(:kw, x, default)`
# (`x=default`) or `Expr(:kw, Expr(:(::), x, T), default)` (`x::T=default`)
# for keyword args with a default value.
function _argsym(a)
    a isa Symbol && return a
    if a isa Expr
        a.head == :(::) && return a.args[1]
        if a.head == :kw
            lhs = a.args[1]
            return lhs isa Symbol ? lhs : _argsym(lhs)
        end
    end
    error("unsupported arg form: $a")
end
_argexpr(a) = _argsym(a)

# An argument's type annotation (`x::T` or `x::T=default`), or `nothing`.
function _annotation(a)
    a isa Expr && a.head == :kw && return _annotation(a.args[1])
    a isa Expr && a.head == :(::) && length(a.args) == 2 && return a.args[2]
    return nothing
end

# The names a `where` clause introduces: `T`, and `T` from `T<:Real`.
_where_names(ps) = Set{Symbol}(p isa Symbol ? p : p.args[1] for p in ps)

# Whether expression `ex` mentions any symbol in `names`.
_mentions(ex::Symbol, names) = ex in names
_mentions(ex::Expr, names) = any(a -> _mentions(a, names), ex.args)
_mentions(_, names) = false

# `::T` and `::T=default` positional arguments have no name, and a model needs
# one for every argument (it is a field of `Model.args` and a parameter of the
# evaluator). Turing models use them for type parameters, e.g.
# `@model function m(x, ::Type{T}=Vector{Float64}) where {T}`; the `where`
# clause still binds `T` from the named argument's type.
function _name_anonymous_args(args)
    return map(enumerate(args)) do (i, a)
        anon(x) = x isa Expr && x.head == :(::) && length(x.args) == 1
        name = Symbol("__pb_arg_", i)
        if anon(a)
            return Expr(:(::), name, a.args[1])
        elseif a isa Expr && a.head == :kw && anon(a.args[1])
            return Expr(:kw, Expr(:(::), name, a.args[1].args[1]), a.args[2])
        end
        return a
    end
end

# The set of the model's own argument names — BOTH positional (`def[:args]`)
# and keyword (`def[:kwargs]`, e.g. `@model function foo(; N::Int) ... end`)
# — e.g. for `@model function foo(y, n) ... end` this is `Set([:y, :n])`.
# Used only to decide, at macro-expansion time, whether a bare-symbol tilde
# LHS *could* be data (passed in as an argument, possibly `missing`) vs is
# definitely a parameter (any other name — e.g. one only ever assigned
# inside the body). Keyword args must be included here for exactly the same
# reason positional ones are: a model like `coinflip(; N::Int)` observing
# `y ~ filldist(Bernoulli(p), N)` needs `N` recognized as "already a
# concrete value", and a model taking data via a keyword arg needs that
# argument's tilde sites (if any) treated as potentially-observed too.
function _argnames(def)
    names = Symbol[]
    for a in get(def, :args, [])
        push!(names, _argsym(a))
    end
    for a in get(def, :kwargs, [])
        push!(names, _argsym(a))
    end
    return Set(names)
end

# Collects every name declared via `z := expr` anywhere in the model body,
# by walking the (pre-rewrite) body Expr tree with a plain recursive scan —
# deliberately NOT `postwalk`-and-replace like the other passes, since this
# is read-only (we're only gathering names here, not producing a new Expr;
# the actual `z := expr` -> `z = expr` rewrite happens separately in
# `_rewrite_walrus`, after this set has already been folded into `argnames`).
# `:=` parses as `Expr(:(:=), lhs, rhs)`, distinct from Julia's `Expr(:(=), ...)`
# plain assignment, so this can't accidentally pick up ordinary `z = expr`.
function _walrus_names(body)
    names = Symbol[]
    MacroTools.postwalk(body) do x
        if x isa Expr && x.head == :(:=) && x.args[1] isa Symbol
            push!(names, x.args[1])
        end
        x  # always return `x` unchanged — this postwalk is used only for its side effect
    end
    return Set(names)
end

# Turns every `z := expr` into a plain `z = expr` — by the time this runs,
# `_walrus_names` has already recorded `z` into `argnames`, so all `:=` needs
# to do at THIS point is stop being special syntax and become a normal
# variable assignment that the rest of the function body can use.
#
# A plain-name `z := expr` also goes through `record_walrus` (tilde.jl), which
# returns `expr` unchanged and, in the two modes that collect values
# (`PriorMode` for `rand`, `FixedMode` for `predict` and the per-draw chain
# evaluation), records it under `z` — this is how `:=` values reach chains, as
# in Turing. On the sampling/gradient path (`EvalMode`) it is the identity and
# compiles away. An indexed LHS (`x[i] := expr`) is a plain assignment, not
# recorded.
function _rewrite_walrus(body)
    return postwalk(body) do x
        if x isa Expr && x.head == :(:=)
            lhs, rhs = x.args
            lhs isa Symbol || return :($(lhs) = $(rhs))
            :($(lhs) = $(PracticalBayes).record_walrus(__mode__, $(Val)($(QuoteNode(lhs))), $(rhs)))
        else
            x
        end
    end
end

# The `:=` names in order of first appearance — what a chain records per draw.
function _walrus_list(body)
    names = Symbol[]
    MacroTools.postwalk(body) do x
        if x isa Expr && x.head == :(:=) && x.args[1] isa Symbol && !(x.args[1] in names)
            push!(names, x.args[1])
        end
        x
    end
    return Tuple(names)
end

"""
    _rewrite_tildes(body, argnames) -> Expr

Walks `body` and rewrites every `~` expression. `argnames` is the set of the
model's positional argument names, used to decide whether a bare-symbol LHS
is potentially-observed (in argnames) or purely a parameter (not in argnames).
"""
function _rewrite_tildes(body, argnames)
    # `postwalk` visits every subexpression bottom-up and replaces it with
    # whatever the closure returns (or leaves it alone via the `else x` arm).
    # `@capture(x, lhs_ ~ rhs_)` is MacroTools pattern matching: it succeeds
    # only when `x` is literally an Expr of the form `lhs ~ rhs`, binding
    # `lhs`/`rhs` to the matched sub-Exprs. This is far more robust than
    # checking `x.head == :call && x.args[1] == :~` by hand.
    return postwalk(body) do x
        if @capture(x, lhs_ ~ rhs_)
            _tilde_expansion(lhs, rhs, argnames)
        elseif @capture(x, lhs_ .~ rhs_)
            _dot_tilde_expansion(lhs, rhs, argnames)
        else
            x  # not a tilde statement — leave it untouched
        end
    end
end

# Rewrites a single `lhs ~ rhs` site. `rhs` is left completely alone (it's
# just the distribution expression, evaluated as-is at eval time — this is
# where e.g. `Normal(mu, sigma)` gets constructed on every call, same as
# DynamicPPL). Two shapes of `lhs` are supported; anything else is a macro
# expansion-time (not runtime) error, so mistakes show up immediately.
function _tilde_expansion(lhs, rhs, argnames)
    if lhs isa Symbol
        # Plain `x ~ D`. Whether `x`'s value comes from the model's own call
        # arguments (`x` itself, which might be `missing`/`nothing` meaning
        # "actually a parameter") or from conditioning (`getcond`) is decided
        # HERE, at macro-expansion time, by simple Set membership — this is
        # what lets the generated code skip DynamicPPL's runtime
        # `inargnames`/context-walk check entirely.
        s = lhs
        valueexpr = s in argnames ? s : :($(PracticalBayes).getcond(__mode__, $(Val)($(QuoteNode(s)))))
        return quote
            ($(s), __acc__) = $(PracticalBayes).tilde(__mode__, $(Val)($(QuoteNode(s))), $(rhs), $(valueexpr), __acc__)
        end
    elseif lhs isa Expr && lhs.head == :ref
        # Indexed `x[i, j, ...] ~ D`: `lhs.args[1]` is the container
        # expression, `lhs.args[2:end]` are the index expressions (as written
        # by the user — could be loop variables, literals, etc). We require
        # the container itself to be a bare Symbol (not e.g. `foo().x[i]`) so
        # that `Val(:x)` unambiguously names one layout slot family.
        container = lhs.args[1]
        idx = lhs.args[2:end]
        container isa Symbol || error("indexed tilde LHS must index a bare variable, got: $lhs")
        s = container
        if s in argnames
            # The container is a model argument (or a `:=` name), so its
            # elements are DATA: `for i in eachindex(y); y[i] ~ Normal(mu, 1); end`
            # observes each `y[i]`. This is the same Set-membership rule plain
            # `y ~ D` and `y .~ D` already use above, decided here at
            # macro-expansion time so the hot path carries no runtime check.
            #
            # Before this branch existed every indexed site went to
            # `tilde_index` (assume-only), so a looped observe silently turned
            # every data element into a parameter AND wrote prior draws over
            # the caller's data array during tracing. The observe path never
            # writes into `container`; see `tilde_index_observe` in tilde.jl.
            return quote
                __acc__ = $(PracticalBayes).tilde_index_observe(
                    __mode__, $(Val)($(QuoteNode(s))), $(container), ($(idx...),), $(rhs), __acc__
                )
            end
        end
        # `tilde_index` both returns the drawn/observed value AND mutates
        # `container[k] = x` itself (see tilde.jl), so the LHS-assignment
        # target here is just a throwaway — we don't need `container[idx...]`
        # on the left because the container is already updated in place.
        return quote
            (_pb_tmp_, __acc__) = $(PracticalBayes).tilde_index(
                __mode__, $(Val)($(QuoteNode(s))), $(container), ($(idx...),), $(rhs), __acc__
            )
        end
    else
        error("unsupported tilde LHS: $lhs")
    end
end

# `y .~ Dbroadcast` — MVP only supports this as an OBSERVE (`y` already bound
# to data); see the big comment block at the top of tilde.jl for why assume
# via `.~` is deliberately unsupported. Note this returns only `__acc__`
# (no drawn value to assign back), unlike the two functions above.
#
# Unlike plain `~`, we can't tell "data" from "parameter" at runtime here by
# checking for `nothing`/`missing` — a `.~` LHS is typically a pre-declared
# container (e.g. `x = Vector{Float64}(undef, n)`), which is never `nothing`
# even when it's meant to hold UNKNOWN values. So instead of a runtime check,
# we reject the ambiguous case statically: `.~` is only allowed when `lhs` is
# one of the model's own argument names (so it's unambiguously either real
# data or `missing`/`nothing` data, exactly like plain `~`'s argnames check).
# Anything else is almost certainly the "assume via .~" pattern we don't
# support, so we fail at macro-expansion time with a clear message rather
# than silently computing a logpdf against uninitialized memory at runtime.
function _dot_tilde_expansion(lhs, rhs, argnames)
    lhs isa Symbol || error("`.~` LHS must be a bare variable naming an existing collection, got: $lhs")
    lhs in argnames || error(
        "`.~` assume (unknown LHS) is not supported: `$lhs` is not one of this model's " *
        "arguments, so it can't be observed data. Use `x[i] ~ dist` for indexed families " *
        "of unknowns, or an array distribution (`product_distribution`, `MvNormal`) via plain `~`.",
    )
    s = lhs
    return quote
        (__acc__) = $(PracticalBayes).tilde_dot(__mode__, $(Val)($(QuoteNode(s))), $(lhs), $(rhs), __acc__)
    end
end

"""
    @addlogprob!(expr)
    @addlogprob!(expr, depends=(:a, :b, ...))

Add `expr` to the log-likelihood. Valid only inside an [`@model`](@ref) body.

Use it for a log-density term that is not a `~` statement.

`depends` lists the model variables `expr` reads. A [`Gibbs`](@ref) HMC block
that owns none of them then skips the term, which cannot change its gradient.
Name every variable `expr` reads: a missing name silently drops a gradient
term. [`check_depends`](@ref) tests the annotations. Without `depends` the term
is always evaluated.

```julia
@model function m(data)
    a ~ Normal()
    b ~ Normal()
    @addlogprob! loglik_a(a, data) depends=(:a,)
    @addlogprob! loglik_b(b, data) depends=(:b,)
end
```
"""
macro addlogprob!(expr)
    return :(error(
        "`@addlogprob!` can only be used inside an `@model` function body " *
        "(it is rewritten at model-compile time, not evaluated directly)",
    ))
end

# Recognizes `@addlogprob!(expr)` / `@addlogprob! expr` anywhere in the model
# body and rewrites it to accumulate directly into the log-likelihood via the
# same `acc_lik` primitive every observe site (`tilde`/`tilde_dot`) already
# uses — see tilde.jl. Matches on the macro NAME only (`x.args[1] ==
# Symbol("@addlogprob!")`), not via `@capture`, since MacroTools doesn't have
# a macrocall-matching helper as convenient as its call-form ones; a
# macrocall's `args` are `(macro_name, __source__::LineNumberNode, actual_args...)`,
# so the expression itself is always `x.args[3]` for the one-argument form
# this macro supports.
# The final component of a (possibly qualified) macro name:
# `@addlogprob!` and `PracticalBayes.@addlogprob!` both -> `Symbol("@addlogprob!")`.
function _macroname(ex)
    ex isa Symbol && return ex
    if ex isa Expr && ex.head == :. && length(ex.args) == 2
        q = ex.args[2]
        q isa QuoteNode && q.value isa Symbol && return q.value
        q isa Symbol && return q
    end
    return nothing
end

function _rewrite_addlogprob(body)
    return postwalk(body) do x
        # The macro name may be written QUALIFIED (`PracticalBayes.@addlogprob!`
        # / `PB.@addlogprob!`), in which case `args[1]` is an `Expr`
        # (`PB.var"@addlogprob!"`) rather than the bare Symbol. Matching only
        # the bare Symbol left the qualified form un-rewritten, so it reached
        # the `@addlogprob!` stub at run time and raised "can only be used
        # inside an @model function body" — inside an @model body. Compare the
        # final component instead, which handles both spellings.
        if x isa Expr && x.head == :macrocall && _macroname(x.args[1]) == Symbol("@addlogprob!")
            # args are (macro_name, LineNumberNode, expr[, depends=(...)])
            args = x.args[3:end]
            isempty(args) && error("`@addlogprob!` takes at least one expression, got: $x")
            expr = args[1]
            depends = _extract_depends(args[2:end], x)
            depends === nothing &&
                return :(__acc__ = $(PracticalBayes).acc_lik(__acc__, $(expr)))
            # Gated form: evaluate the term ONLY when the current evaluation can
            # be affected by it. `addlogprob_needed` folds to a compile-time
            # constant (tilde.jl), so for a block that skips this term the whole
            # branch — and `expr` with it — is eliminated, not merely short-circuited.
            return quote
                if $(PracticalBayes).addlogprob_needed(__mode__, $(depends))
                    __acc__ = $(PracticalBayes).acc_lik(__acc__, $(expr))
                end
            end
        else
            x
        end
    end
end

# Parse the optional `depends=(:a, :b)` argument of `@addlogprob!` into a
# `Val{(:a, :b)}` expression, or `nothing` when absent. Accepts both the
# keyword form (`depends=(...)`, parsed as `:(=)` in macro argument position)
# and a trailing `;`-separated parameters block, so all of
# `@addlogprob! e depends=(:a,)`, `@addlogprob!(e, depends=(:a,))` and
# `@addlogprob!(e; depends=(:a,))` work.
function _extract_depends(rest, x)
    isempty(rest) && return nothing
    # `@addlogprob!(e; depends=(...))` arrives as a single :parameters Expr.
    if length(rest) == 1 && rest[1] isa Expr && rest[1].head == :parameters
        rest = rest[1].args
    end
    length(rest) == 1 || error(
        "`@addlogprob!` takes one expression and an optional `depends=(...)`, got: $x")
    kw = rest[1]
    (kw isa Expr && kw.head in (:(=), :kw) && kw.args[1] === :depends) || error(
        "`@addlogprob!`'s second argument must be `depends=(:a, :b, ...)`, got: $(kw)")
    names = _depends_names(kw.args[2], x)
    return :($(Val)($(QuoteNode(names))))
end

# `depends` must be resolvable at MACRO-EXPANSION time — the whole point is a
# compile-time constant — so only literal symbol tuples are accepted. A runtime
# value would silently degrade into a dynamic branch that defeats the purpose.
function _depends_names(ex, x)
    if ex isa QuoteNode && ex.value isa Symbol
        return (ex.value,)                       # depends=:a
    elseif ex isa Expr && ex.head == :tuple
        out = Symbol[]
        for a in ex.args
            a isa QuoteNode && a.value isa Symbol ||
                error("`@addlogprob!`'s `depends` must be a tuple of literal " *
                      "symbols like `(:a, :b)`, got `$(a)` in: $x")
            push!(out, a.value)
        end
        return Tuple(out)
    end
    error("`@addlogprob!`'s `depends` must be a literal symbol or tuple of " *
          "symbols like `(:a, :b)`, got `$(ex)` in: $x")
end

"""
    _rewrite_returns(body) -> Expr

Rewrites every `return expr` in `body` to `return (expr, __acc__)`, so the
generated evaluator always returns `(retval, acc)`. Does not descend into
nested function definitions.
"""
function _rewrite_returns(body)
    # Same postwalk-and-replace trick as `_rewrite_tildes`. We don't need to
    # special-case nested function/closure definitions here because a nested
    # `function ... return ... end` would introduce its own local `return`,
    # which Julia scopes to that inner function anyway — rewriting it to
    # reference the outer `__acc__` would be wrong, but in practice model
    # bodies don't define nested named functions with their own returns, and
    # if they do the rewritten inner return simply captures the outer
    # `__acc__` by closure, which is still a plain accumulator value, not a
    # bug — just something to be aware of if you see surprising behavior.
    return postwalk(body) do x
        if x isa Expr && x.head == :return
            retexpr = isempty(x.args) ? :(nothing) : x.args[1]
            return :(return ($(retexpr), __acc__))
        else
            x
        end
    end
end

"""
    @static_model function name(args...) ... end

Define a model exactly as [`@model`](@ref) does, but make
[`build_layout`](@ref) default to `static=true`: a layout whose site records are
held in a `Tuple`, as `juliac --trim` needs.

Use it for a model that will be compiled into a standalone binary. Compile time
grows with the number of site records, so write an indexed family as one array
distribution (`x ~ MvNormal(zeros(n), I)`) rather than a loop of `x[i] ~`.

See the [trimming guide](@ref "Compile a model into a standalone binary").
"""
macro static_model(expr)
    # Exactly what `@model` builds, with the static flag set in the model's
    # info (`_model_info`, model.jl), which `build_layout` reads. One shared
    # expansion keeps the two macros from drifting apart.
    return _model_expr(expr; static=true)
end

"""
    prefers_static_layout(f) -> Bool

Whether a model constructor was defined with [`@static_model`](@ref), and so
should default to a tuple-valued `Layout.meta`. Ordinary `@model` definitions
return `false`.
"""
prefers_static_layout(f) = _model_info(f).static
