# Turn each Turing tutorial (a Quarto `index.qmd` in a TuringLang/docs
# checkout) into two runnable scripts:
#
#   <out>/<tutorial>/turing.jl   the tutorial's Julia chunks, unchanged
#   <out>/<tutorial>/pb.jl       the same chunks with RULES applied
#
# Usage: julia port.jl <docs_checkout> <out_dir> [level]
#
# RULES is the list of source changes a reader has to make to run a Turing
# tutorial on PracticalBayes. Each rule carries a level:
#
#   1  mechanical: what any port needs (`using Turing` -> `using
#      PracticalBayes, Distributions`). A tutorial that runs at level 1
#      needs no changes beyond the import line.
#   2  documented API differences, each with a reason. A tutorial that only
#      runs at level 2 runs after the changes listed in its report row.
#
# Porting at level N applies every rule with level <= N, and records which
# rules actually fired in `<tutorial>/rules_applied.txt`, so the report can
# say "in tutorial X, change Y".

# A rule rewrites matching source text. `pattern` is a Regex, `replacement` a
# SubstitutionString (or a function of the match, as for `replace`).
struct Rule
    level::Int
    name::String
    pattern::Regex
    replacement::Any
    reason::String
end

const RULES = Rule[
    # `Turing` anywhere in a `using` list (`using Turing`, `using A, Turing`,
    # `using Turing, B`), but not `using Turing: X`, which names Turing-only
    # API and is left alone so its failure is reported.
    Rule(1, "import", r"^(\s*using\s(?:[^\n:]*?,\s*)?)Turing\b(?=\s*(?:,|;|#|$))"m,
         s"\1PracticalBayes, Distributions",
         "Turing re-exports Distributions; PracticalBayes does not."),
    # Only the whole-variable form. An indexed one (`@varname(m[1])`) has no
    # Symbol equivalent and is left for the report to flag: the PB spelling is
    # `param_draws(chn, :m, 1)`.
    Rule(2, "varname", r"@varname\((\w+)\)", s":\1",
         "PracticalBayes chains are Symbol-keyed (`chain[:p]`, not `chain[@varname(p)]`)."),
    Rule(2, "vnchain", r"\bVNChain\b", "SymChain",
         "PracticalBayes chains are FlexiChains' `SymChain`, not `VNChain`, so annotations " *
         "and imports name `SymChain` (`f(chain::SymChain)`)."),
]
# Rules retired once PracticalBayes itself accepted the Turing form (2026-11-01):
# `setprogress!` (now exported), and `NUTS()` / `NUTS(; adtype=...)` (PB's own
# `NUTS`, src/nuts.jl). A rule belongs here only while PB needs the change.

# One code chunk of a .qmd file, with the source line it starts on and the
# chunk options (`#| key: value` lines) Quarto would read.
struct Chunk
    line::Int
    code::String
    options::Dict{String,String}
end

function extract_chunks(qmd::AbstractString)
    chunks = Chunk[]
    lines = split(read(qmd, String), '\n')
    i = 1
    while i <= length(lines)
        if startswith(lines[i], "```{julia}")
            start = i
            body = String[]
            opts = Dict{String,String}()
            i += 1
            while i <= length(lines) && !startswith(lines[i], "```")
                m = match(r"^#\|\s*([\w-]+):\s*(.*?)\s*$", lines[i])
                m === nothing ? push!(body, lines[i]) : (opts[m[1]] = m[2])
                i += 1
            end
            push!(chunks, Chunk(start, join(body, '\n'), opts))
        end
        i += 1
    end
    return chunks
end

# Quarto's own semantics: `eval: false` chunks are shown but never run, and
# `error: true` chunks are expected to throw (the rendered page shows the
# error). Both have to be honoured or the scripts would not reproduce what
# the rendered tutorial does.
runs(c::Chunk) = get(c.options, "eval", "true") != "false"
expects_error(c::Chunk) = get(c.options, "error", "false") == "true"

# The runner (runner.jl) splits scripts on these marker lines.
marker(k, c::Chunk) = "# ==== CHUNK $k line=$(c.line)" * (expects_error(c) ? " error-ok" : "")

function apply_rules(code::AbstractString, level::Int, fired::Set{String})
    for r in RULES
        r.level <= level || continue
        new = replace(code, r.pattern => r.replacement)
        new == code || push!(fired, r.name)
        code = new
    end
    return code
end

function port_tutorial(qmd, outdir; level=1)
    mkpath(outdir)
    chunks = filter(runs, extract_chunks(qmd))
    fired = Set{String}()
    open(joinpath(outdir, "turing.jl"), "w") do t
        open(joinpath(outdir, "pb.jl"), "w") do p
            for (k, c) in enumerate(chunks)
                println(t, marker(k, c)); println(t, c.code)
                println(p, marker(k, c)); println(p, apply_rules(c.code, level, fired))
            end
        end
    end
    open(joinpath(outdir, "rules_applied.txt"), "w") do io
        for r in RULES
            r.name in fired && println(io, r.level, "\t", r.name, "\t", r.reason)
        end
    end
    return length(chunks), fired
end

function main(docs, out; level=1)
    tutdir = joinpath(docs, "tutorials")
    for name in sort(readdir(tutdir))
        qmd = joinpath(tutdir, name, "index.qmd")
        isfile(qmd) || continue
        n, fired = port_tutorial(qmd, joinpath(out, name); level=level)
        println(rpad(name, 42), lpad(n, 3), " chunks   rules: ", join(sort!(collect(fired)), ", "))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) in (2, 3) || error("usage: julia port.jl <docs_checkout> <out_dir> [level]")
    main(ARGS[1], ARGS[2]; level=length(ARGS) == 3 ? parse(Int, ARGS[3]) : 1)
end
