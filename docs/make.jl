using Documenter
using DocumenterMermaid
using Literate
using AtmosTransport

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
          "METEO_PREPROCESSING.md", "TM5_LOCAL_SETUP.md", "TRANSPORT_COMPARISON.md",
          "TM5_GRID_REFERENCES.md", "REFERENCE_RUN.md", "GITHUB_SETUP.md"]
    src = joinpath(DEV_SRC, "docs", f)
    dst = joinpath(DEV_DST, f)
    if isfile(src)
        cp(src, dst; force=true)
    end
end

# Copy animation GIF into docs/src/assets/ for embedding
const ASSETS_DIR = joinpath(@__DIR__, "src", "assets")
mkpath(ASSETS_DIR)
let gif_name = "column_mean_animation_small.gif"
    gif_dst = joinpath(ASSETS_DIR, gif_name)
    if !isfile(gif_dst)
        alt_src = joinpath(homedir(), "data", "output", "era5_edgar_preprocessed_f32", gif_name)
        if isfile(alt_src)
            cp(alt_src, gif_dst; force=true)
            @info "Copied $gif_name to docs/src/assets/"
        end
    end
end

pages = [
    "Home" => "index.md",
    "Grids" => "grids.md",
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
        "TM5 Local Setup" => "developer/TM5_LOCAL_SETUP.md",
        "Transport Comparison" => "developer/TRANSPORT_COMPARISON.md",
        "TM5 Grid References" => "developer/TM5_GRID_REFERENCES.md",
        "Reference Run" => "developer/REFERENCE_RUN.md",
        "GitHub Setup" => "developer/GITHUB_SETUP.md",
    ],
    "Performance" => [
        "GPU Double Buffering" => "gpu_double_buffering.md",
    ],
    "API Reference" => "api.md",
]

DocMeta.setdocmeta!(AtmosTransport, :DocTestSetup,
    :(using AtmosTransport); recursive=true)

makedocs(;
    sitename = "AtmosTransport.jl",
    authors = "Christian Frankenberg and contributors",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://RemoteSensingTools.github.io/AtmosTransport",
        assets = String[],
        size_threshold = 500 * 1024,
        size_threshold_warn = 300 * 1024,
    ),
    modules = [AtmosTransport],
    pages = pages,
    warnonly = [:cross_references, :missing_docs, :example_block, :linkcheck],
    doctest = false,
)

deploydocs(;
    repo = "github.com/RemoteSensingTools/AtmosTransport.git",
    devbranch = "main",
    push_preview = true,
)
