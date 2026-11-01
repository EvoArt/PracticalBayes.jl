using Documenter
using PracticalBayes

DocMeta.setdocmeta!(PracticalBayes, :DocTestSetup, :(using PracticalBayes, Distributions); recursive=true)

makedocs(
    modules=[PracticalBayes],
    sitename="PracticalBayes.jl",
    format=Documenter.HTML(prettyurls=get(ENV, "CI", "false") == "true"),
    pages=[
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "How-to guides" => [
            "howto/gibbs.md",
            "howto/predictive.md",
            "howto/map_laplace.md",
            "howto/float32.md",
            "howto/gpu.md",
            "howto/trimming.md",
            "howto/from_turing.md",
        ],
        "Reference" => "reference.md",
        "Explanation" => "explanation.md",
    ],
    checkdocs=:exports,
)

deploydocs(
    repo="github.com/EvoArt/PracticalBayes.git",
    devbranch="master",
    devurl="",
    versions=["v#.#", "v#.#.#", "stable"],
)
