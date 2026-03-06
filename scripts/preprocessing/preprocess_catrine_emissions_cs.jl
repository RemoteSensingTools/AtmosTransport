#!/usr/bin/env julia
# ===========================================================================
# CATRINE Emission Preprocessing — NetCDF → Flat Binary on Cubed-Sphere
#
# Converts all 4 CATRINE emission sources from raw NetCDF to flat binary
# on a cubed-sphere grid. Unit conversions use Unitful.jl for correctness.
#
# Sources:
#   1. GridFED fossil CO2  (monthly, 0.1°, kgCO2/month/m²)
#   2. EDGAR SF6           (annual, 0.1°, Tonnes/cell)
#   3. LMDZ CO2 fluxes     (3-hourly, 1°, kgC/m²/s)
#   4. Zhang Rn222         (monthly, 0.5°, kg/m²/s)
#
# Binary layout:
#   [Header]  4096 bytes — JSON metadata (Nc, Nt, species, time_hours, ...)
#   For each snapshot t = 1..Nt:
#     panel_1(Nc×Nc Float32) | panel_2 | ... | panel_6
#
# All outputs in kg/m²/s (except LMDZ: kgCO2/m²/s after C→CO2 conversion).
#
# Usage:
#   julia --project=. scripts/preprocess_catrine_emissions_cs.jl [Nc]
#
# Defaults: Nc = 180 (GEOS-IT C180)
# ===========================================================================

using AtmosTransport
using AtmosTransport.Architectures: CPU
using AtmosTransport.Grids
using AtmosTransport.Grids: HybridSigmaPressure, cell_area
using AtmosTransport.Parameters
using AtmosTransport.Sources
using AtmosTransport.IO: read_geosfp_cs_grid_info
using NCDatasets
using Dates
using Printf
using JSON3
using Unitful

# =====================================================================
# Unitful-based conversion factors (compile-time verified)
# =====================================================================

# GridFED: kgCO2/month/m² → kg/m²/s
const SECONDS_PER_MONTH = Float64(ustrip(u"s", 1u"yr" / 12))

# EDGAR SF6: Tonnes/cell → kg/cell  (then ÷ area and ÷ sec/yr)
const KG_PER_TONNE      = Float64(ustrip(u"kg", 1u"Mg"))             # 1000.0 (1 tonne = 1 Mg)
const SECONDS_PER_YEAR  = Float64(ustrip(u"s", 1u"yr"))              # 3.15576e7

# LMDZ CO2: kgC/m²/s → kgCO2/m²/s
const M_CO2 = 44.01   # g/mol
const M_C   = 12.011  # g/mol
const KGC_TO_KGCO2 = M_CO2 / M_C  # 3.664

# Print verification
println("Unit conversion constants (Unitful-verified):")
@printf("  SECONDS_PER_MONTH = %.1f s  (%.4f days)\n", SECONDS_PER_MONTH, SECONDS_PER_MONTH/86400)
@printf("  SECONDS_PER_YEAR  = %.1f s  (%.4f days)\n", SECONDS_PER_YEAR, SECONDS_PER_YEAR/86400)
@printf("  KG_PER_TONNE      = %.1f\n", KG_PER_TONNE)
@printf("  KGC_TO_KGCO2      = %.4f\n", KGC_TO_KGCO2)

# =====================================================================
# Configuration
# =====================================================================

const FT = Float32
const DEFAULT_HEADER_SIZE = 4096
const LARGE_HEADER_SIZE   = 65536  # for sources with many timesteps (LMDZ: 6088)
const Nc = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 180

# Data paths
const CATRINE_DIR   = expanduser("~/data/AtmosTransport/catrine")
const EMISSIONS_DIR = joinpath(CATRINE_DIR, "Emissions")
const OUTPUT_DIR    = joinpath(CATRINE_DIR, "preprocessed_c$(Nc)")

const GRIDFED_DIR   = joinpath(EMISSIONS_DIR, "gridfed")
const SF6_FILE      = joinpath(EMISSIONS_DIR, "edgar_v8/v8.0_FT2022_GHG_SF6_2022_TOTALS_emi.nc")
const NOAA_FILE     = joinpath(EMISSIONS_DIR, "sf6_gr_gl.txt")
const LMDZ_DIR      = joinpath(EMISSIONS_DIR, "LMDZ_fluxes")
const RN222_DIR     = joinpath(EMISSIONS_DIR, "ZHANG_Rn222")

# GEOS-IT coordinate reference file
const GEOSIT_COORD_FILE = expanduser("~/data/geosit_c180_catrine/20211201/GEOSIT.20211201.CTM_A1.C180.nc")

# =====================================================================
# Binary writer
# =====================================================================

function write_cs_emission_binary(outpath::String, panels_vec::Vector{NTuple{6, Matrix{FT}}},
                                   Nc::Int; source::String="", species::String="",
                                   units::String="kg/m2/s",
                                   time_hours::Vector{Float64}=Float64[],
                                   extra::Dict{String,Any}=Dict{String,Any}())
    Nt = length(panels_vec)
    n_panel = Nc * Nc

    # Auto-size header: use large header for many-timestep sources
    header_size = Nt > 100 ? LARGE_HEADER_SIZE : DEFAULT_HEADER_SIZE

    header = Dict{String,Any}(
        "magic"        => "ECSF",
        "version"      => 2,
        "Nc"           => Nc,
        "n_panels"     => 6,
        "Nt"           => Nt,
        "float_type"   => string(FT),
        "float_bytes"  => sizeof(FT),
        "header_bytes" => header_size,
        "n_per_panel"  => n_panel,
        "units"        => units,
        "source"       => source,
        "species"      => species,
        "time_hours"   => time_hours,
    )
    merge!(header, extra)

    header_json = JSON3.write(header)
    length(header_json) < header_size ||
        error("Header JSON too large ($(length(header_json)) bytes > $header_size)")

    mkpath(dirname(outpath))
    open(outpath, "w") do io
        header_buf = zeros(UInt8, header_size)
        copyto!(header_buf, 1, Vector{UInt8}(header_json), 1, length(header_json))
        write(io, header_buf)
        for t in 1:Nt, p in 1:6
            write(io, vec(panels_vec[t][p]))
        end
    end

    actual = filesize(outpath)
    expected = header_size + Nt * 6 * n_panel * sizeof(FT)
    @info @sprintf("  Written: %s (%.1f MB, Nt=%d)", outpath, actual / 1e6, Nt)
    actual == expected || @warn "Size mismatch: $actual vs expected $expected"
end

# Single-snapshot convenience
function write_cs_emission_binary(outpath::String, panels::NTuple{6, Matrix{FT}},
                                   Nc::Int; kwargs...)
    write_cs_emission_binary(outpath, [panels], Nc; kwargs...)
end

# =====================================================================
# Grid builder
# =====================================================================

function build_grid(Nc::Int)
    @info "Building CubedSphereGrid C$Nc..."
    params = load_parameters(FT)
    pp = params.planet
    vc = HybridSigmaPressure(FT[0, 0], FT[0, 1])
    grid = CubedSphereGrid(CPU(); FT, Nc, vertical=vc,
                            radius=pp.radius, gravity=pp.gravity,
                            reference_pressure=pp.reference_surface_pressure)

    if isfile(GEOSIT_COORD_FILE)
        lons, lats, clons, clats = read_geosfp_cs_grid_info(GEOSIT_COORD_FILE)
        for p in 1:6, j in 1:Nc, i in 1:Nc
            grid.λᶜ[p][i, j] = lons[i, j, p]
            grid.φᶜ[p][i, j] = lats[i, j, p]
        end
        # Overwrite gnomonic areas with GMAO corner-based areas (matches runtime)
        if clons !== nothing && clats !== nothing
            R = Float64(grid.radius)
            gmao_areas = Sources.compute_areas_from_corners(
                Float64.(clons), Float64.(clats), R, Nc)
            for p in 1:6, j in 1:Nc, i in 1:Nc
                grid.Aᶜ[p][i, j] = FT(gmao_areas[p][i, j])
            end
            @info "  Loaded GMAO coordinates + corner-based cell areas from $(basename(GEOSIT_COORD_FILE))"
        else
            @info "  Loaded GMAO coordinates from $(basename(GEOSIT_COORD_FILE)) (no corners — gnomonic areas)"
        end
    else
        @warn "  GEOS-IT coordinate file not found: $GEOSIT_COORD_FILE"
        @warn "  Using gnomonic coordinates — emission panels may be wrong!"
    end
    return grid
end

# =====================================================================
# Helper: regrid + renormalize
# =====================================================================

function regrid_and_renormalize!(flux_native::Matrix{FT}, lon_src, lat_src,
                                  grid::CubedSphereGrid{FT}, cs_map;
                                  R = FT(grid.radius)) where FT
    Nc = grid.Nc
    Nlon_s, Nlat_s = size(flux_native)
    Δlon_s = FT(abs(Float64(lon_src[2]) - Float64(lon_src[1])))
    Δlat_s = FT(abs(Float64(lat_src[2]) - Float64(lat_src[1])))

    # Total mass rate on native grid
    total_native = zero(FT)
    @inbounds for j in 1:Nlat_s, i in 1:Nlon_s
        φ_s = FT(lat_src[j]) - Δlat_s / 2
        φ_n = FT(lat_src[j]) + Δlat_s / 2
        area = R^2 * deg2rad(Δlon_s) * abs(sind(φ_n) - sind(φ_s))
        total_native += flux_native[i, j] * area
    end

    # Conservative regrid
    flux_panels = Sources.regrid_latlon_to_cs(flux_native, FT.(lon_src), FT.(lat_src),
                                                grid; cs_map)

    # Renormalize to preserve global mass
    total_cs = zero(FT)
    for p in 1:6, j in 1:Nc, i in 1:Nc
        total_cs += flux_panels[p][i, j] * cell_area(i, j, grid; panel=p)
    end
    if abs(total_cs) > zero(FT)
        scale = total_native / total_cs
        for p in 1:6
            flux_panels[p] .*= scale
        end
    end

    return flux_panels, total_native
end

# =====================================================================
# 1. GridFED fossil CO2
# =====================================================================

function preprocess_gridfed(grid::CubedSphereGrid{FT}) where FT
    @info "=" ^ 70
    @info "1. GridFED fossil CO2 → C$(grid.Nc) binary"
    @info "=" ^ 70

    isdir(GRIDFED_DIR) || (@warn "GridFED dir not found: $GRIDFED_DIR — skipping"; return)

    nc_files = sort(filter(f -> endswith(f, ".nc"), readdir(GRIDFED_DIR, join=true)))
    isempty(nc_files) && (@warn "No NetCDF files in $GRIDFED_DIR"; return)

    snapshots = NTuple{6, Matrix{FT}}[]
    time_hours = Float64[]

    # Read coordinates from first file, determine if lat needs flipping
    ds0 = NCDataset(nc_files[1])
    lon_src = Float64.(ds0["longitude"][:])
    lat_raw = Float64.(ds0["latitude"][:])
    close(ds0)
    need_flip = lat_raw[1] > lat_raw[end]
    lat_src = need_flip ? reverse(lat_raw) : lat_raw

    @info "  Building regrid map (0.1° → C$(grid.Nc))..."
    cs_map = Sources.build_latlon_to_cs_map(FT.(lon_src), FT.(lat_src), grid)
    @info "  Regrid map built."

    t_offset = 0.0  # hours from simulation start (Dec 1 2021)
    for filepath in nc_files
        ds = NCDataset(filepath)
        units_str = ds["TOTAL"].attrib["units"]
        @info "  $(basename(filepath)): units='$units_str'"

        raw = ds["TOTAL"][:, :, :]
        Nt_file = size(raw, 3)
        close(ds)

        for ti in 1:Nt_file
            slice = FT.(replace(raw[:, :, ti], missing => zero(FT)))

            # Unit conversion: kgCO2/month/m² → kgCO2/m²/s
            slice ./= FT(SECONDS_PER_MONTH)

            # Ensure S→N
            if need_flip
                slice = slice[:, end:-1:1]
            end

            panels, = regrid_and_renormalize!(slice, lon_src, lat_src, grid, cs_map)
            push!(snapshots, panels)
            push!(time_hours, t_offset + (ti - 1) * (SECONDS_PER_MONTH / 3600.0))
        end
        t_offset += Nt_file * (SECONDS_PER_MONTH / 3600.0)
    end

    # Report global total
    Nt = length(snapshots)
    total_GtCO2_yr = sum(sum(snapshots[ti][p] .* FT.(grid.Aᶜ[p]))
                          for p in 1:6 for ti in 1:Nt) / Nt * FT(SECONDS_PER_YEAR) / FT(1e12)
    @info @sprintf("  Global total: %.2f GtCO2/yr (%d monthly snapshots)", Float64(total_GtCO2_yr), Nt)

    outpath = joinpath(OUTPUT_DIR, "gridfed_fossil_co2_cs_c$(grid.Nc)_float32.bin")
    write_cs_emission_binary(outpath, snapshots, grid.Nc;
                              source="GridFED v2024.0 fossil CO2",
                              species="fossil_co2",
                              units="kgCO2/m2/s",
                              time_hours=time_hours)
end

# =====================================================================
# 2. EDGAR SF6
# =====================================================================

function preprocess_edgar_sf6(grid::CubedSphereGrid{FT}) where FT
    @info "=" ^ 70
    @info "2. EDGAR SF6 → C$(grid.Nc) binary"
    @info "=" ^ 70

    isfile(SF6_FILE) || (@warn "EDGAR SF6 not found: $SF6_FILE — skipping"; return)

    ds = NCDataset(SF6_FILE)
    # Find emission variable
    flux_var = nothing
    for name in ["emissions", "emi_sf6", "SF6_emissions", "emi", "TOTALS"]
        if haskey(ds, name)
            flux_var = name
            break
        end
    end
    flux_var === nothing && error("No SF6 emission variable found in $SF6_FILE")

    lon_src = Float64.(ds["lon"][:])
    lat_src = Float64.(ds["lat"][:])
    emi_raw = FT.(replace(ds[flux_var][:, :], missing => zero(FT)))
    units_str = get(ds[flux_var].attrib, "units", "unknown")
    close(ds)
    @info "  EDGAR SF6: $(size(emi_raw)), units='$units_str'"

    # Unit conversion: Tonnes/cell → kg/m²/s
    R = FT(grid.radius)
    Δlon = FT(abs(lon_src[2] - lon_src[1]))
    Δlat = FT(abs(lat_src[2] - lat_src[1]))
    Nlon_e, Nlat_e = length(lon_src), length(lat_src)

    flux_kgm2s = Matrix{FT}(undef, Nlon_e, Nlat_e)
    @inbounds for j in 1:Nlat_e, i in 1:Nlon_e
        φ_s = FT(lat_src[j]) - Δlat / 2
        φ_n = FT(lat_src[j]) + Δlat / 2
        cell_area_e = R^2 * deg2rad(Δlon) * abs(sind(φ_n) - sind(φ_s))
        # Tonnes/yr → kg/m²/s (Unitful-verified constants)
        flux_kgm2s[i, j] = emi_raw[i, j] * FT(KG_PER_TONNE) / (FT(SECONDS_PER_YEAR) * cell_area_e)
    end

    # Ensure S→N
    if lat_src[1] > lat_src[end]
        flux_kgm2s = flux_kgm2s[:, end:-1:1]
        lat_src = reverse(lat_src)
    end

    # Regrid
    @info "  Building regrid map..."
    cs_map = Sources.build_latlon_to_cs_map(FT.(lon_src), FT.(lat_src), grid)
    panels, = regrid_and_renormalize!(flux_kgm2s, lon_src, lat_src, grid, cs_map)

    # NOAA scale factor
    noaa_scale = 1.0
    if isfile(NOAA_FILE)
        noaa_scale = Sources._noaa_sf6_scale_factor(NOAA_FILE, 2022, 2022)
        @info @sprintf("  NOAA scale factor: %.4f", noaa_scale)
    end

    total_kg_yr = Float64(sum(sum(panels[p] .* FT.(grid.Aᶜ[p])) for p in 1:6)) * SECONDS_PER_YEAR
    @info @sprintf("  EDGAR SF6: %.2f kg/yr (before NOAA scaling)", total_kg_yr)

    outpath = joinpath(OUTPUT_DIR, "edgar_sf6_cs_c$(grid.Nc)_float32.bin")
    write_cs_emission_binary(outpath, panels, grid.Nc;
                              source="EDGAR v8.0 SF6 2022",
                              species="sf6",
                              units="kg/m2/s",
                              extra=Dict{String,Any}("noaa_scale" => noaa_scale))
end

# =====================================================================
# 3. LMDZ CO2 fluxes
# =====================================================================

function preprocess_lmdz_co2(grid::CubedSphereGrid{FT};
                              start_date::Date = Date(2021, 12, 1),
                              end_date::Date = Date(2023, 12, 31)) where FT
    @info "=" ^ 70
    @info "3. LMDZ CO2 → C$(grid.Nc) binary"
    @info "=" ^ 70

    isdir(LMDZ_DIR) || (@warn "LMDZ dir not found: $LMDZ_DIR — skipping"; return)

    # Find monthly files in date range
    nc_files = sort(filter(f -> endswith(f, ".nc"), readdir(LMDZ_DIR, join=true)))
    isempty(nc_files) && (@warn "No NetCDF files in $LMDZ_DIR"; return)

    # Filter by date range
    files_in_range = Tuple{Date, String}[]
    for f in nc_files
        m = match(r"(\d{6})", basename(f))
        m === nothing && continue
        file_date = Date(parse(Int, m[1][1:4]), parse(Int, m[1][5:6]), 1)
        if start_date <= file_date <= end_date
            push!(files_in_range, (file_date, f))
        end
    end
    sort!(files_in_range, by=first)
    @info "  Found $(length(files_in_range)) monthly files in date range"

    # Build regrid map from first file
    ds0 = NCDataset(files_in_range[1][2])
    lon_var = haskey(ds0, "longitude") ? "longitude" : "lon"
    lat_var = haskey(ds0, "latitude") ? "latitude" : "lat"
    lon_src = Float64.(ds0[lon_var][:])
    lat_src = Float64.(ds0[lat_var][:])
    close(ds0)

    if lat_src[1] > lat_src[end]
        lat_src = reverse(lat_src)
    end

    @info "  Building LMDZ→C$(grid.Nc) regrid map ($(length(lon_src))×$(length(lat_src)))..."
    cs_map = Sources.build_latlon_to_cs_map(FT.(lon_src), FT.(lat_src), grid)
    @info "  Regrid map built."

    sim_start = DateTime(start_date)
    snapshots = NTuple{6, Matrix{FT}}[]
    time_hours = Float64[]

    for (_, filepath) in files_in_range
        ds = NCDataset(filepath)
        flux_var = "flux_apos"
        haskey(ds, flux_var) || (close(ds); @warn "  No $flux_var in $(basename(filepath))"; continue)

        time_dts = ds["time"][:]
        raw_flux = ds[flux_var]
        Nt_file = length(time_dts)

        lat_file = Float64.(ds[lat_var][:])
        need_flip = lat_file[1] > lat_file[end]

        for ti in 1:Nt_file
            slice = FT.(replace(raw_flux[:, :, ti], missing => zero(FT)))

            # Unit conversion: kgC/m²/s → kgCO2/m²/s (Unitful-verified constant)
            slice .*= FT(KGC_TO_KGCO2)

            if need_flip
                slice = slice[:, end:-1:1]
            end

            panels, = regrid_and_renormalize!(slice, Float64.(lon_src), lat_src, grid, cs_map)
            push!(snapshots, panels)

            hrs = Dates.value(DateTime(time_dts[ti]) - sim_start) / 3_600_000.0
            push!(time_hours, hrs)
        end
        close(ds)
        @info "  $(basename(filepath)): $Nt_file timesteps"
    end

    Nt = length(snapshots)
    @info "  Total: $Nt timesteps from $(length(files_in_range)) files"

    outpath = joinpath(OUTPUT_DIR, "lmdz_co2_cs_c$(grid.Nc)_float32.bin")
    write_cs_emission_binary(outpath, snapshots, grid.Nc;
                              source="LMDZ/CAMS CO2 posterior (flux_apos)",
                              species="co2",
                              units="kgCO2/m2/s",
                              time_hours=time_hours)
end

# =====================================================================
# 4. Zhang Rn222
# =====================================================================

function preprocess_zhang_rn222(grid::CubedSphereGrid{FT}) where FT
    @info "=" ^ 70
    @info "4. Zhang Rn222 → C$(grid.Nc) binary"
    @info "=" ^ 70

    isdir(RN222_DIR) || (@warn "Zhang Rn222 dir not found: $RN222_DIR — skipping"; return)

    nc_files = filter(f -> endswith(f, ".nc"), readdir(RN222_DIR, join=true))
    isempty(nc_files) && (@warn "No NetCDF files in $RN222_DIR"; return)

    ds = NCDataset(nc_files[1])

    # Find variable
    flux_var = nothing
    for name in ["rn_emis", "rnemis", "emis", "Rn222"]
        if haskey(ds, name)
            flux_var = name
            break
        end
    end
    flux_var === nothing && error("No Rn222 variable found in $(nc_files[1])")

    lon_var = haskey(ds, "longitude") ? "longitude" : "lon"
    lat_var = haskey(ds, "latitude") ? "latitude" : "lat"
    lon_src = Float64.(ds[lon_var][:])
    lat_src = Float64.(ds[lat_var][:])
    units_str = get(ds[flux_var].attrib, "units", "kg/m2/s")
    raw = ds[flux_var][:, :, :]
    Nt = size(raw, 3)
    close(ds)
    @info "  Zhang Rn222: $(size(raw)), units='$units_str', $Nt months"

    # Already in kg/m²/s — no conversion needed
    need_flip = lat_src[1] > lat_src[end]
    if need_flip
        lat_src = reverse(lat_src)
    end

    @info "  Building regrid map (0.5° → C$(grid.Nc))..."
    cs_map = Sources.build_latlon_to_cs_map(FT.(lon_src), FT.(lat_src), grid)
    @info "  Regrid map built."

    snapshots = NTuple{6, Matrix{FT}}[]
    time_hours = Float64[]

    for ti in 1:Nt
        slice = FT.(replace(raw[:, :, ti], missing => zero(FT)))
        if need_flip
            slice = slice[:, end:-1:1]
        end

        panels, = regrid_and_renormalize!(slice, lon_src, lat_src, grid, cs_map)
        push!(snapshots, panels)
        push!(time_hours, (ti - 1) * (SECONDS_PER_MONTH / 3600.0))  # monthly
    end

    total_kg_yr = Float64(sum(sum(snapshots[ti][p] .* FT.(grid.Aᶜ[p]))
                                for p in 1:6 for ti in 1:Nt)) / Nt * SECONDS_PER_YEAR
    @info @sprintf("  Zhang Rn222: %.2f kg/yr (%d monthly snapshots)", total_kg_yr, Nt)

    outpath = joinpath(OUTPUT_DIR, "zhang_rn222_cs_c$(grid.Nc)_float32.bin")
    write_cs_emission_binary(outpath, snapshots, grid.Nc;
                              source="Zhang & Liu et al. Rn222",
                              species="rn222",
                              units="kg/m2/s",
                              time_hours=time_hours)
end

# =====================================================================
# Main
# =====================================================================

function main()
    @info "CATRINE emission preprocessing for C$Nc"
    @info "Output directory: $OUTPUT_DIR"
    mkpath(OUTPUT_DIR)

    grid = build_grid(Nc)

    preprocess_gridfed(grid)
    preprocess_edgar_sf6(grid)
    preprocess_lmdz_co2(grid)
    preprocess_zhang_rn222(grid)

    @info "\nAll done. Binary files in: $OUTPUT_DIR"
    for f in sort(readdir(OUTPUT_DIR))
        sz = filesize(joinpath(OUTPUT_DIR, f)) / 1e6
        @info @sprintf("  %s  (%.1f MB)", f, sz)
    end
end

main()
