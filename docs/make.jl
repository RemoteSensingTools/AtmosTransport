using Documenter
using Literate
using AtmosTransportModel

# Process Literate.jl scripts (no execution -- they reference large data files)
const LITERATE_DIR = joinpath(@__DIR__, "literate")
const OUTPUT_DIR = joinpath(@__DIR__, "src", "literated")

literate_scripts = [
    "advection_theory.jl",
    "design_principles.jl",
    "first_forward_run.jl",
    "met_driver_comparison.jl",
]

for script in literate_scripts
    filepath = joinpath(LITERATE_DIR, script)
    if isfile(filepath)
        Literate.markdown(filepath, OUTPUT_DIR;
            flavor = Literate.DocumenterFlavor(),
            execute = false)
    end
end

# Copy developer docs into docs/src/developer/ so Documenter can find them
const DEV_SRC = joinpath(@__DIR__, "..")
const DEV_DST = joinpath(@__DIR__, "src", "developer")
mkpath(DEV_DST)

for f in ["VALIDATION.md", "TM5_CODE_ALIGNMENT.md", "MASS_FLUX_EVOLUTION.md",
          "METEO_PREPROCESSING.md"]
    src = joinpath(DEV_SRC, "docs", f)
    dst = joinpath(DEV_DST, f)
    if isfile(src)
        cp(src, dst; force=true)
    end
end

pages = [
    "Home" => "index.md",
    "Theory" => [
        "Advection Theory" => "literated/advection_theory.md",
        "Design Principles" => "literated/design_principles.md",
    ],
    "Tutorials" => [
        "First Forward Run" => "literated/first_forward_run.md",
        "Met Driver Comparison" => "literated/met_driver_comparison.md",
    ],
    "Developer Guide" => [
        "Validation" => "developer/VALIDATION.md",
        "TM5 Code Alignment" => "developer/TM5_CODE_ALIGNMENT.md",
        "Mass-Flux Evolution" => "developer/MASS_FLUX_EVOLUTION.md",
        "Meteo Preprocessing" => "developer/METEO_PREPROCESSING.md",
    ],
    "API Reference" => "api.md",
]

DocMeta.setdocmeta!(AtmosTransportModel, :DocTestSetup,
    :(using AtmosTransportModel); recursive=true)

makedocs(;
    sitename = "AtmosTransportModel.jl",
    authors = "Christian Frankenberg and contributors",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://RemoteSensingTools.github.io/AtmosTransportModel",
    ),
    modules = [AtmosTransportModel],
    pages = pages,
    warnonly = [:cross_references, :missing_docs],
    doctest = false,
)

deploydocs(;
    repo = "github.com/RemoteSensingTools/AtmosTransportModel.git",
    devbranch = "main",
    push_preview = true,
)
