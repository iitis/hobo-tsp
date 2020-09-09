using Documenter, BinaryOptimization

makedocs(
    modules = [BinaryOptimization],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "Adam Glos",
    sitename = "BinaryOptimization.jl",
    pages = Any["index.md"]
    # strict = true,
    # clean = true,
    # checkdocs = :exports,
)

deploydocs(
    repo = "github.com/aglos@iitis.pl/BinaryOptimization.jl.git",
    push_preview = true
)
