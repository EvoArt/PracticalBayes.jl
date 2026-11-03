# Build the two environments the tutorial scripts run in, from the docs
# checkout's own Project.toml AND Manifest.toml:
#
#   <envs>/turing   exactly the docs' environment, as the tutorials are rendered
#   <envs>/pb       the same environment minus Turing, plus PracticalBayes
#
# Usage: julia setup_envs.jl <docs_checkout> <practicalbayes_path> <envs_dir> [both|turing|pb]
#
# Run this with the Julia version the docs' Manifest was made with (its
# `julia_version` line). The Manifest matters: re-resolving the docs' Project
# from scratch picks up newer releases than the tutorials were written
# against, and the tutorials then fail under Turing itself. (Seen: a newer
# RecursiveArrayTools makes `sol[i]` on an ODE solution a scalar, which broke
# the differential-equations tutorial on the Turing side.) The PB environment
# starts from the same Manifest and lets Pkg move only what PracticalBayes'
# compat bounds force, so the two sides differ as little as possible.
using Pkg

docs, pb, envs = ARGS[1:3]

manifest_julia = match(r"julia_version = \"([^\"]+)\"", read(joinpath(docs, "Manifest.toml"), String))[1]
if VersionNumber(manifest_julia) != VERSION
    @warn "The docs Manifest was made with Julia $manifest_julia, this is $VERSION; Pkg may re-resolve."
end

manifest_versions(dir) = Dict(e.name => e.version
                              for (_, e) in Pkg.Types.read_manifest(joinpath(dir, "Manifest.toml"))
                              if e.version !== nothing)

function make_env(dir; drop=String[], develop=String[])
    mkpath(dir)
    for f in ("Project.toml", "Manifest.toml")
        cp(joinpath(docs, f), joinpath(dir, f); force=true)
    end
    Pkg.activate(dir)
    isempty(drop) || Pkg.rm(drop)
    before = manifest_versions(dir)
    for path in develop
        # Every docs version kept, if at all possible: a PB environment that
        # silently moved packages runs a different tutorial stack from the
        # Turing side. (Seen: PB's old `Bijectors = "0.15.24"` bound forced a
        # re-resolve that moved SciMLBase 2 -> 3 and broke the diffeq tutorial.)
        try
            Pkg.develop(path=path; preserve=Pkg.PRESERVE_ALL)
        catch e
            @warn "PracticalBayes does not fit the docs Manifest as-is; re-resolving" exception = e
            Pkg.develop(path=path)
        end
    end
    after = manifest_versions(dir)
    moved = sort([k for k in keys(before) if haskey(after, k) && after[k] != before[k]])
    if !isempty(moved)
        println("==== versions changed from the docs Manifest:")
        foreach(k -> println("  ", k, "  ", before[k], " -> ", after[k]), moved)
    end
    Pkg.instantiate()
    Pkg.precompile()
    Pkg.status()
end

which = length(ARGS) >= 4 ? ARGS[4] : "both"   # "both", "turing" or "pb"
which in ("both", "turing") && (println("==== turing env"); make_env(joinpath(envs, "turing")))
which in ("both", "pb") && (println("==== pb env"); make_env(joinpath(envs, "pb"); drop=["Turing"], develop=[pb]))
