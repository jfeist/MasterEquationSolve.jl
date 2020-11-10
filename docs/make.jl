using MasterEquationSolve
using Documenter

makedocs(;
    modules=[MasterEquationSolve],
    authors="Johannes Feist <johannes.feist@gmail.com> and contributors",
    repo="https://github.com/jfeist/MasterEquationSolve.jl/blob/{commit}{path}#L{line}",
    sitename="MasterEquationSolve.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jfeist.github.io/MasterEquationSolve.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jfeist/MasterEquationSolve.jl",
)
