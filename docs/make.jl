using Flowfusion
using Documenter

DocMeta.setdocmeta!(Flowfusion, :DocTestSetup, :(using Flowfusion); recursive=true)

makedocs(;
    modules=[Flowfusion],
    authors="murrellb <murrellb@gmail.com> and contributors",
    sitename="Flowfusion.jl",
    format=Documenter.HTML(;
        canonical="https://MurrellGroup.github.io/Flowfusion.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MurrellGroup/Flowfusion.jl",
    devbranch="main",
)
