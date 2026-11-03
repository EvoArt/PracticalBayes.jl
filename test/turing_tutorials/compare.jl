# Read the runner's TOML records for both sides of every tutorial and write a
# Markdown report: does the PracticalBayes port run, where does it first
# fail and why, and do its posteriors agree with Turing's?
#
# Usage: julia compare.jl <work_dir> <report.md>
#
# <work_dir>/<tutorial>/{turing,pb}.toml come from runner.jl and
# <work_dir>/<tutorial>/rules_applied.txt from port.jl.
#
# Posterior agreement: for every chain variable both sides produced, each
# scalar parameter element present in both is compared as
# |mean_pb - mean_turing| / sd_turing. Two correct samplers on the same
# model disagree by Monte Carlo error only, a small fraction of a posterior
# sd; more than 0.25 sd is flagged for a look (label switching in mixture
# models is a known, legitimate cause).
import TOML

const TOL_SD = 0.25

function load(path)
    isfile(path) || return nothing
    return TOML.parsefile(path)
end

status_word(r) = r === nothing ? "not run" :
                 r["first_failure"] == 0 ? "runs" : "fails at chunk $(r["first_failure"])"

function first_error(r)
    r === nothing && return ""
    r["first_failure"] == 0 && return ""
    c = r["chunks"][r["first_failure"]]
    return first(split(c["error"], "\n--- top frames ---")[1], 300)
end

function chain_agreement(t, p)
    (t === nothing || p === nothing) && return (0, 0, String[])
    ncompared, nflagged, worst = 0, 0, String[]
    for (cname, tparams) in t["chains"]
        (tparams isa AbstractDict && haskey(p["chains"], cname)) || continue
        pparams = p["chains"][cname]
        pparams isa AbstractDict || continue
        for (k, ts) in tparams
            haskey(pparams, k) || continue
            ps = pparams[k]
            (ts isa AbstractDict && ps isa AbstractDict) || continue
            (isfinite(ts["mean"]) && isfinite(ps["mean"])) || continue
            sd = ts["std"] > 0 ? ts["std"] : 1.0
            z = abs(ps["mean"] - ts["mean"]) / sd
            ncompared += 1
            if z > TOL_SD
                nflagged += 1
                push!(worst, "$cname.$k: turing $(round(ts["mean"]; sigdigits=4)) vs pb $(round(ps["mean"]; sigdigits=4)) ($(round(z; digits=2)) sd)")
            end
        end
    end
    return (ncompared, nflagged, worst)
end

function main(work, report)
    io = IOBuffer()
    println(io, "| tutorial | Turing | PracticalBayes | rules applied | params compared | > $(TOL_SD) sd |")
    println(io, "|---|---|---|---|---|---|")
    details = IOBuffer()
    for name in sort(filter(d -> isdir(joinpath(work, d)), readdir(work)))
        t = load(joinpath(work, name, "turing.toml"))
        p = load(joinpath(work, name, "pb.toml"))
        rules_path = joinpath(work, name, "rules_applied.txt")
        rules = isfile(rules_path) ? join((split(l, '\t')[2] for l in eachline(rules_path)), ", ") : ""
        n, nf, worst = chain_agreement(t, p)
        println(io, "| $name | $(status_word(t)) | $(status_word(p)) | $rules | $n | $nf |")
        if p !== nothing && p["first_failure"] != 0 || !isempty(worst) || (t !== nothing && t["first_failure"] != 0)
            println(details, "\n### $name\n")
            if t !== nothing && t["first_failure"] != 0
                println(details, "Turing itself fails at chunk $(t["first_failure"]):\n\n```\n$(first_error(t))\n```\n")
            end
            if p !== nothing && p["first_failure"] != 0
                println(details, "PracticalBayes first fails at chunk $(p["first_failure"]) ($(p["chunks"][p["first_failure"]]["header"])):\n\n```\n$(first_error(p))\n```\n")
                later = [c for c in p["chunks"] if c["status"] == "error" && c["after_first_failure"]]
                isempty(later) || println(details, "Later failing chunks: ", join((c["chunk"] for c in later), ", "), "\n")
            end
            isempty(worst) || println(details, "Posterior disagreements:\n\n", join(("- " * w for w in worst[1:min(end, 10)]), "\n"), "\n")
        end
    end
    write(report, String(take!(io)) * "\n## Details\n" * String(take!(details)))
    println(read(report, String))
end

main(ARGS[1], ARGS[2])
