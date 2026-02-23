# ---------------------------------------------------------------------------
# PreprocessedLatLonMetDriver — reads pre-computed mass fluxes from binary/NetCDF
#
# Mass fluxes (m, am, bm, cm, ps) were pre-computed by
# preprocess_mass_fluxes.jl. This driver reads them directly — no wind
# staggering or pressure computation needed.
#
# Supports two file formats:
#   .bin — mmap'd flat binary via MassFluxBinaryReader (fast, preferred)
#   .nc  — NetCDF4 random-access
#
# Supports monthly-sharded files: set `directory` and file discovery
# chains through shards in chronological order.
#
# Extracted from: scripts/run_forward_preprocessed.jl
# ---------------------------------------------------------------------------

using NCDatasets

"""
$(TYPEDEF)

Met driver for pre-computed lat-lon mass fluxes (binary or NetCDF).

$(FIELDS)
"""
struct PreprocessedLatLonMetDriver{FT} <: AbstractMassFluxMetDriver{FT}
    "ordered list of mass-flux file paths (.bin or .nc)"
    files           :: Vector{String}
    "windows per file (indexed by file position)"
    windows_per_file :: Vector{Int}
    "total number of met windows across all files"
    n_windows       :: Int
    "advection sub-step size [s]"
    dt              :: FT
    "number of advection sub-steps per met window"
    steps_per_win   :: Int
    "longitude vector"
    lons            :: Vector{FT}
    "latitude vector"
    lats            :: Vector{FT}
    Nx              :: Int
    Ny              :: Int
    Nz              :: Int
    "topmost model level index"
    level_top       :: Int
    "bottommost model level index"
    level_bot       :: Int
end

"""
    PreprocessedLatLonMetDriver(; FT, files, dt=nothing)

Construct a preprocessed lat-lon met driver from file list.
Grid metadata and dt are read from the first file's header.
If `dt` is provided, it overrides the file's embedded value.
"""
function PreprocessedLatLonMetDriver(; FT::Type{<:AbstractFloat} = Float64,
                                       files::Vector{String},
                                       dt::Union{Nothing, Real} = nothing)
    isempty(files) && error("PreprocessedLatLonMetDriver: no files provided")

    # Read metadata from first file
    r = MassFluxBinaryReader(files[1], FT)
    Nx, Ny, Nz = r.Nx, r.Ny, r.Nz
    file_dt = r.dt_seconds
    steps_per = r.steps_per_met
    lons = copy(r.lons)
    lats = copy(r.lats)
    level_top = r.level_top
    level_bot = r.level_bot
    close(r)

    actual_dt = dt === nothing ? file_dt : FT(dt)
    steps_per_win = dt === nothing ? steps_per : max(1, round(Int, actual_dt * steps_per / file_dt))

    # Count windows per file
    wins_per = Int[]
    total = 0
    for f in files
        r2 = MassFluxBinaryReader(f, FT)
        push!(wins_per, r2.Nt)
        total += r2.Nt
        close(r2)
    end

    PreprocessedLatLonMetDriver{FT}(
        files, wins_per, total,
        FT(actual_dt), steps_per_win,
        lons, lats, Nx, Ny, Nz,
        level_top, level_bot)
end

# --- Interface implementations ---

total_windows(d::PreprocessedLatLonMetDriver)    = d.n_windows
window_dt(d::PreprocessedLatLonMetDriver)        = d.dt * d.steps_per_win
steps_per_window(d::PreprocessedLatLonMetDriver) = d.steps_per_win

"""
    window_to_file_local(driver, win) → (file_idx, local_win)

Map a global window index to (file index, within-file window index).
"""
function window_to_file_local(d::PreprocessedLatLonMetDriver, win::Int)
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
    load_met_window!(cpu_buf::LatLonCPUBuffer, driver::PreprocessedLatLonMetDriver, grid, win)

Read pre-computed mass fluxes for window `win` into `cpu_buf`.
Opens the appropriate file, reads the window, and closes.
"""
function load_met_window!(cpu_buf::LatLonCPUBuffer,
                           driver::PreprocessedLatLonMetDriver{FT},
                           grid, win::Int) where FT
    file_idx, local_win = window_to_file_local(driver, win)
    filepath = driver.files[file_idx]

    if endswith(filepath, ".bin")
        reader = MassFluxBinaryReader(filepath, FT)
        load_window!(cpu_buf.m, cpu_buf.am, cpu_buf.bm, cpu_buf.cm, cpu_buf.ps,
                     reader, local_win)
        close(reader)
    else
        # NetCDF fallback
        ds = NCDataset(filepath, "r")
        try
            cpu_buf.m  .= FT.(ds["m"][:, :, :, local_win])
            cpu_buf.am .= FT.(ds["am"][:, :, :, local_win])
            cpu_buf.bm .= FT.(ds["bm"][:, :, :, local_win])
            cpu_buf.cm .= FT.(ds["cm"][:, :, :, local_win])
            cpu_buf.ps .= FT.(ds["ps"][:, :, local_win])
        finally
            close(ds)
        end
    end
    return nothing
end
