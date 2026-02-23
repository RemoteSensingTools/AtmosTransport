# ---------------------------------------------------------------------------
# ERA5MetDriver — reads raw ERA5 model-level winds, computes mass fluxes
#
# This driver reads cell-centered u/v winds + log surface pressure from
# ERA5 NetCDF files, staggers winds, builds pressure thickness, and
# computes mass fluxes on the fly (no preprocessed binary needed).
#
# Extracted from: scripts/run_era5_edgar_double_buffer.jl
# ---------------------------------------------------------------------------

using NCDatasets
using Dates

"""
$(TYPEDEF)

Met driver that reads raw ERA5 model-level winds and computes mass fluxes
on the fly. Suitable for lat-lon grids.

Wind fields are read from NetCDF, interpolated to staggered (face) positions,
and mass fluxes are computed from pressure thickness + staggered winds.

$(FIELDS)
"""
struct ERA5MetDriver{FT} <: AbstractRawMetDriver{FT}
    "ordered list of ERA5 NetCDF file paths (daily files)"
    files          :: Vector{String}
    "hybrid A coefficients for the level range (Nz+1 values)"
    A_coeff        :: Vector{FT}
    "hybrid B coefficients for the level range (Nz+1 values)"
    B_coeff        :: Vector{FT}
    "time between met updates [s]"
    met_interval   :: FT
    "advection sub-step size [s]"
    dt             :: FT
    "number of advection sub-steps per met window"
    steps_per_win  :: Int
    "number of timesteps per file"
    nt_per_file    :: Int
    "total number of met windows"
    n_windows      :: Int
    "longitude vector (from first file)"
    lons           :: Vector{FT}
    "latitude vector (from first file, S→N)"
    lats           :: Vector{FT}
    "topmost model level index"
    level_top      :: Int
    "bottommost model level index"
    level_bot      :: Int
end

"""
    ERA5MetDriver(; FT, files, A_coeff, B_coeff, met_interval=21600, dt=900,
                    level_top=50, level_bot=137)

Construct an ERA5 met driver from a list of NetCDF file paths and hybrid
coefficients. Grid metadata (lons, lats, Nt) is read from the first file.
"""
function ERA5MetDriver(; FT::Type{<:AbstractFloat} = Float64,
                         files::Vector{String},
                         A_coeff::Vector{<:Real},
                         B_coeff::Vector{<:Real},
                         met_interval::Real = 21600,
                         dt::Real = 900,
                         level_top::Int = 50,
                         level_bot::Int = 137)
    isempty(files) && error("ERA5MetDriver: no files provided")
    lons, lats, _, _, _, _, nt_per_file = get_era5_grid_info(files[1], FT)
    steps_per_win = max(1, round(Int, met_interval / dt))
    n_windows = nt_per_file * length(files)

    ERA5MetDriver{FT}(
        files,
        FT.(A_coeff), FT.(B_coeff),
        FT(met_interval), FT(dt), steps_per_win,
        nt_per_file, n_windows,
        lons, lats, level_top, level_bot)
end

# --- Interface implementations ---

total_windows(d::ERA5MetDriver)    = d.n_windows
window_dt(d::ERA5MetDriver)        = d.dt * d.steps_per_win
steps_per_window(d::ERA5MetDriver) = d.steps_per_win
met_interval(d::ERA5MetDriver)     = d.met_interval

"""
    window_index_to_file(driver::ERA5MetDriver, win) → (filepath, tidx)

Map a global window index to (file path, within-file time index).
"""
function window_index_to_file(d::ERA5MetDriver, win::Int)
    file_idx = div(win - 1, d.nt_per_file) + 1
    tidx     = mod1(win, d.nt_per_file)
    return d.files[file_idx], tidx
end

"""
    load_met_window!(cpu_buf::LatLonCPUBuffer, driver::ERA5MetDriver, grid, win)

Read ERA5 winds for window `win`, stagger them, and build pressure thickness
into `cpu_buf`. Returns the surface pressure array.

For ERA5, the CPU buffer is filled with:
  - `cpu_buf.am` ← staggered u-wind (Nx+1, Ny, Nz)
  - `cpu_buf.bm` ← staggered v-wind (Nx, Ny+1, Nz)
  - `cpu_buf.m`  ← pressure thickness (Nx, Ny, Nz)
  - `cpu_buf.ps` ← surface pressure (Nx, Ny)

Note: For ERA5, the CPU buffer fields `am/bm` temporarily hold staggered
winds (not mass fluxes). The GPU-side compute step converts them to true
mass fluxes via `compute_mass_fluxes!`.
"""
function load_met_window!(cpu_buf::LatLonCPUBuffer, driver::ERA5MetDriver{FT},
                           grid, win::Int) where FT
    filepath, tidx = window_index_to_file(driver, win)
    u_cc, v_cc, ps = load_era5_timestep(filepath, tidx, FT)
    Nx, Ny, Nz = size(cpu_buf.m)

    # Stagger winds from cell-center to faces
    stagger_winds!(cpu_buf.am, cpu_buf.bm, u_cc, v_cc, Nx, Ny, Nz)

    # Build pressure thickness Δp on the CPU
    # Using the grid's vertical coordinate A/B coefficients
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        ΔA = driver.A_coeff[k + 1] - driver.A_coeff[k]
        ΔB = driver.B_coeff[k + 1] - driver.B_coeff[k]
        cpu_buf.m[i, j, k] = ΔA + ΔB * ps[i, j]
    end

    copyto!(cpu_buf.ps, ps)
    return nothing
end
