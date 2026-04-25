# ===========================================================================
# TOML-driven source-descriptor factory.
#
# `load_met_settings(toml_path; root_dir, kwargs...)` reads a met_sources
# TOML and returns the typed `AbstractMetSettings` for that source. Dispatch
# is on `[source].name` — currently "GEOS-IT" and "GEOS-FP" map to the
# corresponding `GEOSSettings{flavor}` aliases. MERRA-2 (LL) and other
# sources will plug in by extending the `name => Settings` mapping.
# ===========================================================================

"""
    load_met_settings(toml_path::String; root_dir, kwargs...) -> AbstractMetSettings

Construct a typed met-source descriptor from `toml_path`. The TOML's
`[source].name` key picks the concrete settings type. The `[preprocessing]`
table (if present) supplies defaults for `level_orientation`,
`mass_flux_dt_seconds`, and `include_convection`; explicit keyword
arguments override.

`root_dir` is the on-disk directory holding the source's daily NetCDF files
(e.g. `~/data/AtmosTransport/met/geosit/C180/raw_catrine`).
"""
function load_met_settings(toml_path::String;
                           root_dir::AbstractString,
                           kwargs...)
    isfile(toml_path) || error("Met source TOML not found: $toml_path")
    cfg = TOML.parsefile(toml_path)
    name = cfg["source"]["name"]

    grid_cfg     = get(cfg, "grid",     Dict{String,Any}())
    vertical_cfg = get(cfg, "vertical", Dict{String,Any}())
    pre_cfg      = get(cfg, "preprocessing", Dict{String,Any}())

    coefs = String(get(vertical_cfg, "coefficients_file", "config/geos_L72_coefficients.toml"))

    if name == "GEOS-IT" || name == "GEOS-FP"
        Nc = Int(grid_cfg["Nc"])
        mass_flux_dt = Float64(get(pre_cfg, "mass_flux_dt_seconds", 450.0))
        level_orientation = Symbol(get(pre_cfg, "level_orientation", "auto"))
        include_convection = Bool(get(pre_cfg, "include_convection", false))

        ctor = name == "GEOS-IT" ? GEOSITSettings : GEOSFPSettings
        return ctor(; root_dir = String(root_dir),
                      Nc, mass_flux_dt, level_orientation,
                      include_convection, coefficients_file = coefs, kwargs...)
    end

    error("Unsupported met source `$(name)`. Supported: GEOS-IT, GEOS-FP.")
end
