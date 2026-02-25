# ---------------------------------------------------------------------------
# GEOSFPCubedSphereMetDriver — GEOS-FP mass fluxes on cubed-sphere panels
#
# Two ingestion modes:
#   :binary — preprocessed flat binary via CSBinaryReader (fast, mmap)
#   :netcdf — raw GEOS-FP NetCDF files via read_geosfp_cs_timestep
#
# Extracted from: scripts/run_forward_geosfp_cs_gpu.jl
# ---------------------------------------------------------------------------

using Dates

"""
$(TYPEDEF)

Met driver for GEOS-FP cubed-sphere mass fluxes.

$(FIELDS)
"""
struct GEOSFPCubedSphereMetDriver{FT} <: AbstractMassFluxMetDriver{FT}
    "ordered list of data file paths (.bin or .nc4)"
    files           :: Vector{String}
    "ingestion mode: :binary or :netcdf"
    mode            :: Symbol
    "windows per file (for binary multi-file mode)"
    windows_per_file :: Vector{Int}
    "total number of met windows"
    n_windows       :: Int
    "cells per panel edge"
    Nc              :: Int
    "number of vertical levels"
    Nz              :: Int
    "halo width"
    Hp              :: Int
    "time between met updates [s]"
    met_interval    :: FT
    "advection sub-step size [s]"
    dt              :: FT
    "number of advection sub-steps per met window"
    steps_per_win   :: Int
    "NetCDF file for reading panel coordinates (used by binary mode for regridding)"
    coord_file      :: String
    "merge map for vertical level merging (native level → merged level index), or nothing"
    merge_map       :: Union{Nothing, Vector{Int}}
    "accumulation time for mass fluxes [s] (dynamics timestep; defaults to met_interval)"
    mass_flux_dt    :: FT
end

"""
    GEOSFPCubedSphereMetDriver(; FT, preprocessed_dir="", netcdf_files=[],
                                  start_date, end_date, dt=900, met_interval=3600, Hp=3)

Construct a GEOS-FP cubed-sphere met driver.

If `preprocessed_dir` is given and contains `.bin` files, uses binary mode.
Otherwise falls back to NetCDF files (either from `netcdf_files` or
discovered in a data directory).
"""
function GEOSFPCubedSphereMetDriver(;
        FT::Type{<:AbstractFloat} = Float32,
        preprocessed_dir::String = "",
        netcdf_files::Vector{String} = String[],
        start_date::Date = Date("2024-06-01"),
        end_date::Date = Date("2024-06-05"),
        dt::Real = 900,
        met_interval::Real = 3600,
        Hp::Int = 3,
        merge_map::Union{Nothing, Vector{Int}} = nothing,
        mass_flux_dt::Real = met_interval)

    ft_tag = FT == Float32 ? "float32" : "float64"
    steps_per_win = max(1, round(Int, met_interval / dt))

    if !isempty(preprocessed_dir) && isdir(preprocessed_dir)
        # Binary mode
        bin_files = find_preprocessed_cs_files(preprocessed_dir, start_date, end_date, ft_tag)
        isempty(bin_files) && error("No preprocessed .bin files found in $preprocessed_dir")

        wins_per = Int[]
        Nc = 0; Nz_file = 0; Hp_file = 0
        for f in bin_files
            r = CSBinaryReader(f, FT)
            push!(wins_per, r.Nt)
            Nc = r.Nc; Nz_file = r.Nz; Hp_file = r.Hp
            close(r)
        end
        total = sum(wins_per)

        # Use first NetCDF file (if available) for panel coordinate reading
        coord_file = isempty(netcdf_files) ? "" : netcdf_files[1]

        merge_map !== nothing &&
            error("Binary mode with layer merging not yet supported. Use netcdf_dir instead.")

        return GEOSFPCubedSphereMetDriver{FT}(
            bin_files, :binary, wins_per, total,
            Nc, Nz_file, Hp_file,
            FT(met_interval), FT(dt), steps_per_win,
            coord_file, nothing, FT(mass_flux_dt))
    else
        # NetCDF mode
        files = isempty(netcdf_files) ? String[] : netcdf_files
        isempty(files) && error(
            "GEOSFPCubedSphereMetDriver: no preprocessed_dir or netcdf_files provided")

        # Read grid info from first file
        ts0 = read_geosfp_cs_timestep(files[1]; FT)
        Nc = ts0.Nc
        Nz_file = ts0.Nz

        # Probe time dimension: daily files have 24 timesteps, hourly files have 1
        wins_per = Int[]
        for f in files
            nt = NCDataset(f, "r") do ds
                length(ds["time"])
            end
            push!(wins_per, nt)
        end
        n_windows = sum(wins_per)

        return GEOSFPCubedSphereMetDriver{FT}(
            files, :netcdf, wins_per, n_windows,
            Nc, Nz_file, Hp,
            FT(met_interval), FT(dt), steps_per_win,
            files[1],  # coord_file = first NetCDF file
            merge_map, FT(mass_flux_dt))
    end
end

# --- Interface implementations ---

total_windows(d::GEOSFPCubedSphereMetDriver)    = d.n_windows
window_dt(d::GEOSFPCubedSphereMetDriver)        = d.dt * d.steps_per_win
steps_per_window(d::GEOSFPCubedSphereMetDriver) = d.steps_per_win
met_interval(d::GEOSFPCubedSphereMetDriver)     = d.met_interval

"""
    window_to_file_local(driver::GEOSFPCubedSphereMetDriver, win) → (file_idx, local_win)

Map a global window index to (file index, within-file window index).
"""
function window_to_file_local(d::GEOSFPCubedSphereMetDriver, win::Int)
    cumul = 0
    for (fi, nw) in enumerate(d.windows_per_file)
        if win <= cumul + nw
            return fi, win - cumul
        end
        cumul += nw
    end
    error("Window index $win out of range (total: $(d.n_windows))")
end

"""
    load_met_window!(cpu_buf::CubedSphereCPUBuffer, driver::GEOSFPCubedSphereMetDriver,
                      grid, win)

Read cubed-sphere met fields for window `win` into CPU staging buffer.

In binary mode, reads directly from mmap'd binary via CSBinaryReader.
In NetCDF mode, reads raw GEOS-FP data and converts to staggered format.
"""
function load_met_window!(cpu_buf::CubedSphereCPUBuffer,
                           driver::GEOSFPCubedSphereMetDriver{FT},
                           grid, win::Int) where FT
    file_idx, local_win = window_to_file_local(driver, win)
    filepath = driver.files[file_idx]

    @info "  Met window $win: $(basename(filepath)) [t=$local_win]" maxlog=200

    if driver.mode === :binary
        cached_path = ensure_local_cache(filepath)
        reader = CSBinaryReader(cached_path, FT)
        load_cs_window!(cpu_buf.delp, cpu_buf.am, cpu_buf.bm, reader, local_win)
        close(reader)
    else
        # NetCDF mode — read, halo, and stagger
        ts = read_geosfp_cs_timestep(filepath; FT, time_index=local_win,
                                      dt_met=driver.mass_flux_dt,
                                      convert_to_kgs=true)
        delp_haloed, mfxc, mfyc = to_haloed_panels(ts; Hp=driver.Hp)
        am_stag, bm_stag = cgrid_to_staggered_panels(mfxc, mfyc)

        mm = driver.merge_map
        if mm !== nothing
            # Regrid native levels → merged levels by summing within groups
            for p in 1:6
                cpu_buf.delp[p] .= zero(FT)
                cpu_buf.am[p]   .= zero(FT)
                cpu_buf.bm[p]   .= zero(FT)
                for k in 1:length(mm)
                    km = mm[k]
                    cpu_buf.delp[p][:, :, km] .+= delp_haloed[p][:, :, k]
                    cpu_buf.am[p][:, :, km]   .+= am_stag[p][:, :, k]
                    cpu_buf.bm[p][:, :, km]   .+= bm_stag[p][:, :, k]
                end
            end
        else
            for p in 1:6
                copyto!(cpu_buf.delp[p], delp_haloed[p])
                copyto!(cpu_buf.am[p], am_stag[p])
                copyto!(cpu_buf.bm[p], bm_stag[p])
            end
        end
    end
    return nothing
end
