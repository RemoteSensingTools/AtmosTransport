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
using Printf
using Statistics: mean

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
    "print field statistics (wind speed, DELP ordering, NaN) on first window and every 24th"
    verbose         :: Bool
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
        coord_file::String = "",
        start_date::Date = Date("2024-06-01"),
        end_date::Date = Date("2024-06-05"),
        dt::Real = 900,
        met_interval::Real = 3600,
        Hp::Int = 3,
        merge_map::Union{Nothing, Vector{Int}} = nothing,
        mass_flux_dt::Real = met_interval,
        verbose::Bool = false)

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

        # Use explicit coord_file kwarg, else first NetCDF file, for panel coordinate reading
        cfile = !isempty(coord_file) ? coord_file :
                !isempty(netcdf_files) ? netcdf_files[1] : ""

        merge_map !== nothing &&
            error("Binary mode with layer merging not yet supported. Use netcdf_dir instead.")

        return GEOSFPCubedSphereMetDriver{FT}(
            bin_files, :binary, wins_per, total,
            Nc, Nz_file, Hp_file,
            FT(met_interval), FT(dt), steps_per_win,
            cfile, nothing, FT(mass_flux_dt), verbose)
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
            merge_map, FT(mass_flux_dt), verbose)
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
Derive the A3mstE file path from a CTM_A1 file path by replacing the collection
name in the filename. Returns empty string if the file doesn't exist.
"""
function _a3mste_path_from_ctm(ctm_path::String)
    dir = dirname(ctm_path)
    fname = basename(ctm_path)
    a3_fname = replace(fname, "CTM_A1" => "A3mstE")
    a3_path = joinpath(dir, a3_fname)
    return isfile(a3_path) ? a3_path : ""
end

"""
    load_cmfmc_window!(cmfmc_panels, driver::GEOSFPCubedSphereMetDriver, grid, win)

Load convective mass flux (CMFMC) from A3mstE files into pre-allocated panels.

CMFMC data is 3-hourly (8 timesteps/day) while CTM_A1 is hourly (24/day).
Each A3mstE timestep covers 3 hourly windows.

Returns `true` if CMFMC was loaded, `false` if A3mstE data is not available.
"""
function load_cmfmc_window!(cmfmc_panels::NTuple{6},
                             driver::GEOSFPCubedSphereMetDriver{FT},
                             grid, win::Int) where FT
    driver.mode === :netcdf || return false  # binary mode: no CMFMC support yet

    file_idx, local_win = window_to_file_local(driver, win)
    ctm_path = driver.files[file_idx]

    # Find the A3mstE file alongside the CTM_A1 file
    a3_path = _a3mste_path_from_ctm(ctm_path)
    isempty(a3_path) && return false

    # Map hourly CTM_A1 window → 3-hourly A3mstE timestep
    # CTM_A1 local_win 1-3 → A3mstE t=1, 4-6 → t=2, ..., 22-24 → t=8
    a3_time_index = cld(local_win, 3)  # ceiling division

    cmfmc_haloed = read_geosfp_cs_cmfmc(a3_path; FT, time_index=a3_time_index,
                                         Hp=driver.Hp)

    # Handle vertical level merging if active
    mm = driver.merge_map
    if mm !== nothing
        Nz_merged = maximum(mm)
        Nz_edge_merged = Nz_merged + 1
        for p in 1:6
            fill!(cmfmc_panels[p], zero(FT))
            # For edge-level data, merge by taking the flux at the merged interface
            # The topmost and bottommost edges stay zero. Inner edges map via the
            # merge map: interface k sits between layers mm[k-1] and mm[k].
            # Use the max flux within each merged group as a conservative estimate.
            Nz_edge_native = size(cmfmc_haloed[p], 3)
            Nc_h = size(cmfmc_panels[p], 1)
            for k_native in 1:Nz_edge_native
                # Map native edge k → merged edge
                # Edge k sits between layer k-1 and layer k
                # For k=1 (TOA) and k=Nz_edge (surface), always 0
                if k_native == 1
                    k_merged = 1
                elseif k_native == Nz_edge_native
                    k_merged = Nz_edge_merged
                else
                    k_merged = mm[k_native - 1] + 1  # edge below merged layer
                end
                for jj in 1:Nc_h, ii in 1:Nc_h
                    cmfmc_panels[p][ii, jj, k_merged] =
                        max(cmfmc_panels[p][ii, jj, k_merged],
                            cmfmc_haloed[p][ii, jj, k_native])
                end
            end
        end
    else
        for p in 1:6
            copyto!(cmfmc_panels[p], cmfmc_haloed[p])
        end
    end

    return true
end

"""
Derive the A1 file path from a CTM_A1 file path by replacing the collection
name in the filename. Returns empty string if the file doesn't exist.
"""
function _a1_path_from_ctm(ctm_path::String)
    dir = dirname(ctm_path)
    fname = basename(ctm_path)
    a1_fname = replace(fname, "CTM_A1" => "A1")
    a1_path = joinpath(dir, a1_fname)
    return isfile(a1_path) ? a1_path : ""
end

"""
    load_surface_fields_window!(sfc_panels, driver::GEOSFPCubedSphereMetDriver, grid, win)

Load surface fields (PBLH, USTAR, HFLUX, T2M) from A1 files into pre-allocated panels.

A1 data is hourly (24 timesteps/day), same as CTM_A1, so the time index maps directly.

`sfc_panels` is a NamedTuple `(pblh, ustar, hflux, t2m)` where each is an
NTuple{6} of 2D arrays (Nc+2Hp, Nc+2Hp).

Returns `true` if surface fields were loaded, `false` if A1 data is not available.
"""
function load_surface_fields_window!(sfc_panels::NamedTuple,
                                      driver::GEOSFPCubedSphereMetDriver{FT},
                                      grid, win::Int) where FT
    driver.mode === :netcdf || return false

    file_idx, local_win = window_to_file_local(driver, win)
    ctm_path = driver.files[file_idx]

    a1_path = _a1_path_from_ctm(ctm_path)
    isempty(a1_path) && return false

    fields = read_geosfp_cs_surface_fields(a1_path; FT, time_index=local_win,
                                            Hp=driver.Hp)

    for p in 1:6
        copyto!(sfc_panels.pblh[p],  fields.pblh[p])
        copyto!(sfc_panels.ustar[p], fields.ustar[p])
        copyto!(sfc_panels.hflux[p], fields.hflux[p])
        copyto!(sfc_panels.t2m[p],   fields.t2m[p])
    end

    return true
end

# ---------------------------------------------------------------------------
# Verbose sanity checks
#
# Called on window 1 and every 24th window when driver.verbose = true.
# Checks for:
#   - NaN / Inf contamination
#   - Column pressure sum (~1e5 Pa expected)
#   - Inverted vertical ordering (DELP[k=1] should be thin, stratospheric)
#   - Plausible wind speed from mass flux (catches wrong mass_flux_dt)
# ---------------------------------------------------------------------------

const _GRAV_CS = 9.80616f0
const _R_EARTH_CS = 6.371e6

function _sanity_check_cs_buf(delp::NTuple{6}, am::NTuple{6}, bm::NTuple{6},
                                win::Int, mass_flux_dt::Real)
    p_delp = delp[1]   # (Nc+2Hp, Nc+2Hp, Nz)
    p_am   = am[1]     # (Nc+1, Nc, Nz)

    Nc = size(p_am, 2)
    dy = 2π * _R_EARTH_CS / (4 * Nc)   # approximate cell edge length [m]

    Hp = (size(p_delp, 1) - size(p_am, 1) + 1)   # halo width
    inner = p_delp[Hp:end-Hp+1, Hp:end-Hp+1, :]  # strip halo

    n_nan = count(isnan, inner) + count(isnan, p_am) + count(isnan, bm[1])
    n_inf = count(isinf, inner) + count(isinf, p_am) + count(isinf, bm[1])
    if n_nan > 0 || n_inf > 0
        @warn "[met sanity win=$win] NaN=$n_nan, Inf=$n_inf — data corruption!"
    end

    # Column pressure sum: sum(DELP) over levels ≈ surface pressure ~1e5 Pa
    ps_mean = mean(sum(inner, dims=3))
    if !(8e4 < ps_mean < 1.1e5)
        msg = @sprintf("[met sanity win=%d] Column DELP sum = %.0f Pa (expected ~1e5 Pa). Check units or vertical ordering.", win, ps_mean)
        @warn msg
    end

    # Vertical ordering: DELP at level 1 should be thin (stratosphere ≈ few Pa)
    delp_top = mean(inner[:, :, 1])
    delp_bot = mean(inner[:, :, end])
    if delp_top > delp_bot
        msg = @sprintf("[met sanity win=%d] DELP[k=1]=%.2f > DELP[k=end]=%.1f Pa — levels appear inverted (bottom-to-top). Check auto-detection.", win, delp_top, delp_bot)
        @warn msg
    end

    # Estimated surface wind: am is total mass flux [kg/s] through the cell face.
    # u = am * g / (DELP * dy)  where dy is the cell edge length.
    am_rms   = sqrt(mean(x -> x^2, p_am[:, :, end]))
    u_est    = am_rms * _GRAV_CS / (delp_bot * dy)
    if u_est > 80.0 && win == 1
        msg = @sprintf("[met sanity win=%d] Estimated |u_sfc| ≈ %.1f m/s — suspiciously high! Check mass_flux_dt (currently %.0fs).", win, u_est, mass_flux_dt)
        @warn msg
    elseif u_est < 0.5 && win == 1
        msg = @sprintf("[met sanity win=%d] Estimated |u_sfc| ≈ %.3f m/s — too low! Check mass_flux_dt (currently %.0fs).", win, u_est, mass_flux_dt)
        @warn msg
    end

    msg = @sprintf("[met sanity win=%d] ps≈%.0fPa | DELP top=%.2f bot=%.1f Pa | est |u_sfc|=%.1f m/s (dy=%.0fm) | mass_flux_dt=%.0fs", win, ps_mean, delp_top, delp_bot, u_est, dy, mass_flux_dt)
    @info msg
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

    # Per-window logging removed — progress bar in run loop handles this

    if driver.mode === :binary
        cached_path = ensure_local_cache(filepath)
        reader = CSBinaryReader(cached_path, FT)
        load_cs_window!(cpu_buf.delp, cpu_buf.am, cpu_buf.bm, reader, local_win)
        close(reader)
    else  # :netcdf
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

    if driver.verbose && (win == 1 || win % 24 == 0)
        _sanity_check_cs_buf(cpu_buf.delp, cpu_buf.am, cpu_buf.bm,
                              win, driver.mass_flux_dt)
    end
    return nothing
end
