using Documenter
using DocumenterVitepress    # VitePress frontend; native Mermaid via VitePress
using Literate               # .jl tutorial source → .md (with executed output)
using AtmosTransport

# ---------------------------------------------------------------------------
# Literate.jl preprocessing — every .jl file under docs/literate/ becomes a
# Markdown page under docs/src/tutorials/. Source is the canonical file;
# the generated .md files are rebuilt from scratch each Documenter run, so
# they live under docs/src/tutorials/_generated/ which is git-ignored.
# ---------------------------------------------------------------------------

const LITERATE_SRC = joinpath(@__DIR__, "literate")
const LITERATE_OUT = joinpath(@__DIR__, "src", "tutorials", "_generated")

if isdir(LITERATE_SRC)
    rm(LITERATE_OUT; force = true, recursive = true)
    mkpath(LITERATE_OUT)
    for jl in readdir(LITERATE_SRC; join = true)
        endswith(jl, ".jl") || continue
        Literate.markdown(jl, LITERATE_OUT;
                          flavor  = Literate.DocumenterFlavor(),
                          execute = true,
                          credit  = false)
    end
end

DocMeta.setdocmeta!(AtmosTransport, :DocTestSetup, :(using AtmosTransport);
                    recursive = true)

const PAGES = [
    "Home" => "index.md",
    "Getting Started" => [
        "getting_started/installation.md",
        "getting_started/quickstart.md",
        "getting_started/first_run.md",
        "getting_started/inspecting_output.md",
    ],
    "For TM5 & GCHP users" => [
        "for_tm5_gchp_users/philosophy.md",
        "for_tm5_gchp_users/binary_pipeline.md",
        "for_tm5_gchp_users/operators_on_binaries.md",
        "for_tm5_gchp_users/adjoints.md",
        "for_tm5_gchp_users/kernel_architecture.md",
    ],
    "Concepts" => [
        "concepts/grids.md",
        "concepts/state_and_basis.md",
        "concepts/operators.md",
        "concepts/binary_format.md",
    ],
    "Tutorials" => [
        "tutorials/_generated/synthetic_latlon.md",
    ],
    "Preprocessing" => [
        "preprocessing/overview.md",
        "preprocessing/spectral_era5.md",
        "preprocessing/geos_native_cs.md",
        "preprocessing/regridding.md",
        "preprocessing/conventions.md",
    ],
    "Theory & Verification" => [
        "theory/mass_conservation.md",
        "theory/advection_schemes.md",
        "theory/conservation_budgets.md",
        "theory/validation_status.md",
        "theory/adjoint_status.md",
    ],
    "Configuration & Runtime" => [
        "config/toml_schema.md",
        "config/output_schema.md",
        "config/data_sources.md",
    ],
    "API Reference" => [
        "api/public_api.md",
        "api/index.md",
        "api/architectures.md",
        "api/parameters.md",
        "api/grids.md",
        "api/state.md",
        "api/met_drivers.md",
        "api/operators.md",
        "api/models.md",
        "api/preprocessing.md",
    ],
    "About these docs" => "about.md",
]

makedocs(
    root     = @__DIR__,
    modules  = [AtmosTransport],
    sitename = "AtmosTransport.jl",
    authors  = "RemoteSensingTools and contributors",
    repo     = Remotes.GitHub("RemoteSensingTools", "AtmosTransport.jl"),
    format   = DocumenterVitepress.MarkdownVitepress(
        repo          = "github.com/RemoteSensingTools/AtmosTransport.jl",
        devbranch     = "main",
        devurl        = "dev",
    ),
    pages    = PAGES,
    # Phase 1 keeps the build permissive so missing-docstring / autodoc work
    # in later phases is the trigger for stricter gates, not the infrastructure
    # commit itself.
    warnonly = true,
    checkdocs = :none,
)

# DocumenterVitepress builds separate VitePress outputs under docs/build/1,
# docs/build/2, ... and records their destination folders in docs/build/bases.txt.
# Its deploy wrapper publishes those rendered folders into gh-pages/dev,
# gh-pages/stable, etc. Calling Documenter.deploydocs directly would publish the
# wrapper directory itself and leave the public /dev/ URL without an index page.
const ATMOSTR_DOCS_BUILD_ONLY =
    lowercase(get(ENV, "ATMOSTR_DOCS_BUILD_ONLY", "false")) in ("1", "true", "yes")

if ATMOSTR_DOCS_BUILD_ONLY
    @info "Skipping docs deployment because ATMOSTR_DOCS_BUILD_ONLY is set"
else
    DocumenterVitepress.deploydocs(
        root         = @__DIR__,
        repo         = "github.com/RemoteSensingTools/AtmosTransport.jl.git",
        target       = "build",
        devbranch    = "main",
        branch       = "gh-pages",
        push_preview = true,
        forcepush    = true,
    )
end
