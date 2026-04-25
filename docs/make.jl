using Documenter
using DocumenterMermaid     # Mermaid block rendering in Documenter HTML
using AtmosTransport

# ---------------------------------------------------------------------------
# Phase 1 (infrastructure) — Documenter wiring with a minimal page tree.
#
# Subsequent phases will fill in content under the placeholders that are
# already enumerated here in the proposed structure (commented out below).
# Keeping the live `pages` set tight makes the site green from the first
# commit; pages get uncommented as their content lands.
# ---------------------------------------------------------------------------

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
    "Concepts" => [
        "concepts/grids.md",
        "concepts/state_and_basis.md",
        "concepts/operators.md",
        "concepts/binary_format.md",
    ],
    "About these docs" => "about.md",
]

# Pages tree planned for Phases 2–9 (uncomment as each phase lands):
#
# const PAGES = [
#     "Home" => "index.md",
#     "Getting Started" => [
#         "getting_started/installation.md",
#         "getting_started/first_run.md",
#         "getting_started/inspecting_output.md",
#     ],
#     "Concepts" => [
#         "concepts/grids.md",
#         "concepts/state_and_basis.md",
#         "concepts/operators.md",
#         "concepts/binary_format.md",
#     ],
#     "Theory & Verification" => [
#         "theory/mass_conservation.md",
#         "theory/advection_schemes.md",
#         "theory/conservation_budgets.md",
#         "theory/validation_status.md",
#         "theory/adjoint_status.md",
#     ],
#     "Tutorials" => [
#         "tutorials/latlon_era5.md",
#         "tutorials/cubed_sphere_geosit.md",
#         "tutorials/reduced_gaussian.md",
#     ],
#     "Preprocessing" => [
#         "preprocessing/overview.md",
#         "preprocessing/spectral_era5.md",
#         "preprocessing/geos_native_cs.md",
#         "preprocessing/regridding.md",
#         "preprocessing/conventions.md",
#     ],
#     "Configuration & Runtime" => [
#         "config/toml_schema.md",
#         "config/sample_data.md",
#         "config/output_schema.md",
#         "config/credentials.md",
#     ],
#     "API Reference" => [
#         "api/grids.md",
#         "api/state.md",
#         "api/met_drivers.md",
#         "api/operators.md",
#         "api/models.md",
#         "api/preprocessing.md",
#     ],
#     "Developer Internals" => [
#         "internals/module_layout.md",
#         "internals/runtime_dispatch.md",
#         "internals/gpu_portability.md",
#         "internals/contributing.md",
#     ],
#     "About these docs" => "about.md",
# ]

makedocs(
    modules  = [AtmosTransport],
    sitename = "AtmosTransport.jl",
    authors  = "RemoteSensingTools and contributors",
    repo     = Remotes.GitHub("RemoteSensingTools", "AtmosTransport"),
    format   = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical  = "https://RemoteSensingTools.github.io/AtmosTransport/dev/",
        edit_link  = "main",
        assets     = String[],
    ),
    pages    = PAGES,
    # Phase 1 keeps the build permissive so missing-docstring / autodoc work
    # in later phases is the trigger for stricter gates, not the infrastructure
    # commit itself.
    warnonly = true,
    checkdocs = :none,
)

deploydocs(
    repo         = "github.com/RemoteSensingTools/AtmosTransport.git",
    devbranch    = "main",
    push_preview = true,
    forcepush    = true,
)
