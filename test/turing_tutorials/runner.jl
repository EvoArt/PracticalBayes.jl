# Run one script written by port.jl, chunk by chunk, the way Quarto renders a
# tutorial, and write a TOML record of what happened.
#
# Usage: julia --project=<env> runner.jl <script.jl> <out.toml> [workdir]
#
# Every chunk is evaluated in `Main` (as Quarto's Julia engine does), and its
# value is rendered when Quarto would show it (the chunk's last line does not
# end in `;`). Rendering matters: a plot recipe or a chain's `show` method
# only runs at display time, so skipping it would hide real failures. A
# failing chunk does not stop the run — later chunks are still attempted, so
# one broken plot does not hide whether the sampling after it works — but
# failures after the first are marked, since many will be knock-on errors.
#
# At the end, every global holding a FlexiChain is summarised (posterior mean
# and standard deviation of each scalar parameter element), so compare.jl can
# check a PracticalBayes run against the Turing run of the same tutorial.

ENV["GKSwstype"] = "100"   # GR (Plots' default backend) without a display

# The runner's own names live in this module, not in `Main`, so they cannot
# collide with whatever a tutorial defines (`results`, `render`, ...).
module __TutRunner

import TOML  # stdlib: loadable from any environment, unlike a JSON package

function split_chunks(src)
    chunks = Tuple{String,Bool,String}[]   # (header, error_ok, code)
    header, buf = "", String[]
    for line in split(src, '\n')
        if startswith(line, "# ==== CHUNK ")
            isempty(header) || push!(chunks, (header, occursin("error-ok", header), join(buf, '\n')))
            header, buf = line, String[]
        else
            push!(buf, line)
        end
    end
    isempty(header) || push!(chunks, (header, occursin("error-ok", header), join(buf, '\n')))
    return chunks
end

# Quarto shows a chunk's value unless its last non-blank line ends with `;`.
function shows_value(code)
    lines = filter(!isempty, strip.(split(code, '\n')))
    isempty(lines) && return false
    return !endswith(lines[end], ";")
end

function render(v)
    v === nothing && return
    io = IOBuffer()
    if showable(MIME("image/png"), v)
        show(io, MIME("image/png"), v)
    else
        show(IOContext(io, :limit => true), MIME("text/plain"), v)
    end
    return
end

# --- chain summaries -------------------------------------------------------
# FlexiChains is found among the loaded modules rather than imported, so the
# runner works whether or not the tutorial loaded it.
function flexichains_module()
    for (id, m) in Base.loaded_modules
        id.name == "FlexiChains" && return m
    end
    return nothing
end

function flatten_draws!(out, name, draws)
    vals = vec(collect(draws))
    isempty(vals) && return
    x1 = first(vals)
    if x1 isa Real
        xs = Float64.(vals)
        out[name] = Dict("mean" => sum(xs) / length(xs), "std" => std_(xs), "n" => length(xs))
    elseif x1 isa AbstractArray{<:Real} && all(v -> size(v) == size(x1), vals)
        for I in CartesianIndices(x1)
            idx = join(Tuple(I), ",")
            flatten_draws!(out, "$name[$idx]", [v[I] for v in vals])
        end
    end
    return
end
std_(xs) = length(xs) < 2 ? 0.0 : sqrt(sum(abs2, xs .- sum(xs) / length(xs)) / (length(xs) - 1))

function summarise_chains()
    FC = flexichains_module()
    FC === nothing && return Dict{String,Any}()
    out = Dict{String,Any}()
    for g in names(Main; all=true)
        isdefined(Main, g) || continue
        v = try getfield(Main, g) catch; continue end
        v isa FC.FlexiChain || continue
        params = Dict{String,Any}()
        try
            for (k, draws) in getfield(v, :_data)
                k isa FC.Parameter || continue
                flatten_draws!(params, string(k.name), draws)
            end
        catch e
            params["__error__"] = sprint(showerror, e)
        end
        out[string(g)] = params
    end
    return out
end

# Showing an exception can itself throw (seen: a DomainError holding a
# ForwardDiff Dual, whose printing hit `size(::Partials)`), and an error
# escaping the catch block would lose the record of the whole run. Fall back
# to the exception's type.
function describe(e, bt)
    msg = try
        first(sprint(showerror, e), 1500)
    catch e2
        "$(typeof(e)) (showing it failed with $(typeof(e2)))"
    end
    frames = try
        join((string(f) for f in stacktrace(bt)[1:min(end, 8)]), "\n")
    catch
        "(unavailable)"
    end
    return msg * "\n--- top frames ---\n" * frames
end

# --- main ------------------------------------------------------------------
function main(script, out, workdir)
    workdir === nothing || cd(workdir)
    results = Dict{String,Any}[]
    first_failure = 0
    t_total = @elapsed for (k, (header, error_ok, code)) in enumerate(split_chunks(read(script, String)))
        println(">>> ", header); flush(stdout)
        status, msg, t = "ok", "", 0.0
        try
            t = @elapsed begin
                v = Base.include_string(Main, code, "chunk$k")
                shows_value(code) && Base.invokelatest(render, v)
            end
            error_ok && (status = "unexpected-success")
        catch e
            if error_ok
                status = "expected-error"
            else
                status = "error"
                msg = describe(e, catch_backtrace())
                first_failure == 0 && (first_failure = k)
            end
        end
        status == "error" && println("    ERROR: ", first(msg, 400))
        push!(results, Dict("chunk" => k, "header" => header, "status" => status, "seconds" => t,
                            "after_first_failure" => first_failure != 0 && k > first_failure, "error" => msg))
        flush(stdout)
    end

    chains = try Base.invokelatest(summarise_chains) catch e; Dict("__error__" => sprint(showerror, e)) end
    open(out, "w") do io
        TOML.print(io, Dict("script" => script, "julia" => string(VERSION), "seconds" => t_total,
                            "first_failure" => first_failure, "chunks" => results, "chains" => chains))
    end
    println("DONE first_failure=", first_failure, "  total ", round(t_total; digits=1), " s")
end

end # module

__TutRunner.main(abspath(ARGS[1]), abspath(ARGS[2]), length(ARGS) >= 3 ? ARGS[3] : nothing)
