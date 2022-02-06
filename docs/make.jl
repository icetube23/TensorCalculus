using TensorCalculus
using Documenter

DocMeta.setdocmeta!(TensorCalculus, :DocTestSetup, :(using TensorCalculus); recursive=true)

makedocs(;
    modules=[TensorCalculus],
    authors="Elias Kempf <eli2008@gmx.de> and contributors",
    repo="https://github.com/icetube23/TensorCalculus.jl/blob/{commit}{path}#{line}",
    sitename="TensorCalculus.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://icetube23.github.io/TensorCalculus.jl",
        assets=String[],
        edit_link="main",
    ),
    pages=["Home" => "index.md"],
    strict=:doctest,
)

deploydocs(; repo="github.com/icetube23/TensorCalculus.jl", devbranch="main")
