using Documenter
using MemPalace

makedocs(;
    sitename = "MemPalace.jl",
    modules = [MemPalace],
    remotes = nothing,
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://juliaknowledge.github.io/MemPalace.jl",
        edit_link = "main",
        repolink = "https://github.com/JuliaKnowledge/MemPalace.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Guide" => [
            "Getting Started" => "guide/getting_started.md",
        ],
        "API Reference" => "api.md",
    ],
    warnonly = [:missing_docs, :cross_references, :docs_block],
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/JuliaKnowledge/MemPalace.jl.git",
    devbranch = "main",
    push_preview = true,
)
