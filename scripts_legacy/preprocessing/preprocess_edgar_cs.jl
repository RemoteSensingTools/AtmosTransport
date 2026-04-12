#!/usr/bin/env julia
# ===========================================================================
# One-time EDGAR → Cubed-Sphere Preprocessing
#
# Reads EDGAR v8.0 CO2 emissions (0.1° lat-lon, Tonnes/yr), converts to
# kg/m²/s, regrids to cubed-sphere panels via nearest-neighbor using
# GEOS-FP file coordinates, and saves to a compact binary file.
#
# Binary layout:
#   [Header]  4096 bytes — JSON metadata
#   flux_panel_1..6     — each Nc×Nc Float32 [kg/m²/s]
#
# Usage:
#   julia --project=. scripts/preprocess_edgar_cs.jl
#
# Environment variables:
#   EDGAR_FILE     — input EDGAR NetCDF
#   OUTFILE        — output binary path
#   NC_GRID        — grid resolution (default: 720, for C720)
#   GEOSFP_REF_FILE — any GEOS-FP NetCDF for coordinate reference
# ===========================================================================

using AtmosTransport
using AtmosTransport.Parameters
using AtmosTransport.IO: read_geosfp_cs_grid_info
using NCDatasets
using Printf
using JSON3

const FT = Float32
const NC_GRID = parse(Int, get(ENV, "NC_GRID", "720"))

const EDGAR_FILE = expanduser(get(ENV, "EDGAR_FILE",
    joinpath(homedir(), "data", "emissions", "edgar_v8",
             "v8.0_FT2022_GHG_CO2_2022_TOTALS_emi.nc")))
const OUTFILE = expanduser(get(ENV, "OUTFILE",
    joinpath(homedir(), "data", "emissions", "edgar_v8",
             "edgar_cs_c$(NC_GRID)_float32.bin")))

# Any GEOS-FP NetCDF file — only used for grid coordinates (lons, lats)
const GEOSFP_REF_FILE = expanduser(get(ENV, "GEOSFP_REF_FILE",
    joinpath(homedir(), "data", "geosfp_cs", "20240601",
             "GEOS.fp.asm.tavg_1hr_ctm_c0720_v72.20240601_0030.V01.nc4")))

const HEADER_SIZE = 4096

function regrid_edgar_to_cs(edgar_raw, edgar_lons, edgar_lats,
                            grid_lons, grid_lats, Nc, radius)
    Δlon = edgar_lons[2] - edgar_lons[1]
    Δlat = edgar_lats[2] - edgar_lats[1]
    sec_per_yr = FT(365.25 * 24 * 3600)
    R = FT(radius)
    Nlon_e = length(edgar_lons)
    Nlat_e = length(edgar_lats)

    flux_kgm2s = Matrix{FT}(undef, Nlon_e, Nlat_e)
    @inbounds for j in 1:Nlat_e, i in 1:Nlon_e
        φ_s = FT(edgar_lats[j]) - Δlat / 2
        φ_n = FT(edgar_lats[j]) + Δlat / 2
        cell_area_e = R^2 * deg2rad(Δlon) * abs(sind(φ_n) - sind(φ_s))
        flux_kgm2s[i, j] = FT(edgar_raw[i, j]) * FT(1000) / (sec_per_yr * cell_area_e)
    end

    flux_panels = ntuple(6) do p
        pf = zeros(FT, Nc, Nc)
        for j in 1:Nc, i in 1:Nc
            lon = mod(FT(grid_lons[i, j, p]) + FT(180), FT(360)) - FT(180)
            lat = FT(grid_lats[i, j, p])
            ii = clamp(round(Int, (lon - edgar_lons[1]) / Δlon) + 1, 1, Nlon_e)
            jj = clamp(round(Int, (lat - edgar_lats[1]) / Δlat) + 1, 1, Nlat_e)
            pf[i, j] = flux_kgm2s[ii, jj]
        end
        pf
    end

    total_edgar = sum(edgar_raw) * 1000 / sec_per_yr
    total_cs = sum(1:6) do p
        sum(flux_panels[p])
    end
    @info @sprintf("  EDGAR total: %.1f kg/s, CS total flux sum: %.1f",
                   total_edgar, total_cs)

    return flux_panels
end

function main()
    @info "=" ^ 70
    @info "EDGAR → Cubed-Sphere C$(NC_GRID) Preprocessing (GEOS-FP coords)"
    @info "=" ^ 70

    isfile(EDGAR_FILE) || error("EDGAR file not found: $EDGAR_FILE")
    isfile(GEOSFP_REF_FILE) || error("GEOS-FP reference file not found: $GEOSFP_REF_FILE")

    @info "  Reading EDGAR: $EDGAR_FILE"
    ds = NCDataset(EDGAR_FILE)
    edgar_lons = FT.(ds["lon"][:])
    edgar_lats = FT.(ds["lat"][:])
    edgar_raw  = FT.(replace(ds["emissions"][:, :], missing => zero(FT)))
    close(ds)
    @info @sprintf("  EDGAR grid: %d × %d", length(edgar_lons), length(edgar_lats))

    @info "  Reading GEOS-FP coordinates from: $GEOSFP_REF_FILE"
    file_lons, file_lats, _, _ = read_geosfp_cs_grid_info(GEOSFP_REF_FILE)
    Nc_file = size(file_lons, 1)
    @info "  GEOS-FP grid: C$Nc_file ($(size(file_lons)))"
    Nc_file == NC_GRID || error("GEOS-FP file is C$Nc_file but expected C$NC_GRID")

    params = load_parameters(FT)
    pp = params.planet

    @info "  Regridding to C$(NC_GRID) panels (nearest-neighbor, GEOS-FP coords)..."
    t0 = time()
    flux_panels = regrid_edgar_to_cs(edgar_raw, edgar_lons, edgar_lats,
                                      file_lons, file_lats, NC_GRID, pp.radius)
    @info @sprintf("  Regrid done in %.1fs", time() - t0)

    n_panel = NC_GRID * NC_GRID
    header = Dict{String,Any}(
        "magic"        => "ECSF",
        "version"      => 1,
        "Nc"           => NC_GRID,
        "n_panels"     => 6,
        "float_type"   => "Float32",
        "float_bytes"  => 4,
        "header_bytes" => HEADER_SIZE,
        "n_per_panel"  => n_panel,
        "units"        => "kg/m2/s",
        "source"       => "EDGAR v8.0 CO2 2022 TOTALS",
        "coords"       => "geosfp_file",
    )
    header_json = JSON3.write(header)

    mkpath(dirname(OUTFILE))
    @info "  Writing: $OUTFILE"
    open(OUTFILE, "w") do io
        header_buf = zeros(UInt8, HEADER_SIZE)
        copyto!(header_buf, 1, Vector{UInt8}(header_json), 1, length(header_json))
        write(io, header_buf)
        for p in 1:6
            write(io, vec(flux_panels[p]))
        end
    end

    actual = filesize(OUTFILE)
    expected = HEADER_SIZE + 6 * n_panel * sizeof(FT)
    @info @sprintf("  Done: %.1f MB (expected %.1f MB)", actual / 1e6, expected / 1e6)
    actual == expected || @warn "Size mismatch"
    @info "=" ^ 70
end

main()
