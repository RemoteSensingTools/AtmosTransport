#!/usr/bin/env julia
# =========================================================================
# Per-sweep diagnostic: run 2 hours (2 windows) saving 3D fields after
# each X, Y, Z sweep within the Strang split X→Y→Z→Z→Y→X.
#
# Saves a rectangular SH subregion (j=30:180, lat ≈ -75° to -0.25°)
# where all lat bins are full resolution (no reduced grid).
#
# Output: NetCDF with dimensions (lon, lat_sub, lev, sweep, substep, window)
# Fields: rm (tracer mass), m (air mass), am, bm, cm (mass fluxes)
#
# Usage:
#   julia --threads=2 --project=. scripts/diagnostics/run_2hour_sweep_diag.jl
# =========================================================================

using CUDA
using AtmosTransport
using AtmosTransport.IO: MassFluxBinaryReader, load_window!, LatLonCPUBuffer,
    load_flux_delta_window!
using AtmosTransport.Advection: allocate_massflux_workspace,
    advect_x_massflux_subcycled!, advect_y_massflux_subcycled!,
    advect_z_massflux_subcycled!
using AtmosTransport.Models: _get_cluster_sizes_cpu
using NCDatasets

# ── Configuration ────────────────────────────────────────────────────
const DATA_DIR  = "/temp1/atmos_transport/era5_v4_5000Pa"
const IC_FILE   = expanduser("~/data/AtmosTransport/catrine/InitialConditions/startCO2_202112010000.nc")
const OUT_FILE  = "/tmp/era5_sweep_diag_2hr.nc"
const N_WINDOWS = 2
const N_SUB     = 4       # substeps per window (= steps_per_met_window)
const FT        = Float32

# SH subregion (all rows have cluster_size=1, so rectangular)
const J_RANGE   = 30:180  # lat ≈ -75.25° to -0.25°
const NJ_SUB    = length(J_RANGE)

# Sweep labels: X1 Y1 Z1 Z2 Y2 X2
const SWEEP_NAMES = ["X1", "Y1", "Z1", "Z2", "Y2", "X2"]
const N_SWEEPS = 6

# ── Load met data ────────────────────────────────────────────────────
println("Loading met data...")
files = sort([joinpath(DATA_DIR, f) for f in readdir(DATA_DIR) if endswith(f, ".bin")])
r = MassFluxBinaryReader(files[1], FT)
Nx, Ny, Nz = r.Nx, r.Ny, r.Nz
println("Grid: $(Nx)×$(Ny)×$(Nz), $(r.Nt) windows/file")

# Allocate CPU buffers
m_cpu  = Array{FT}(undef, Nx, Ny, Nz)
am_cpu = Array{FT}(undef, Nx+1, Ny, Nz)
bm_cpu = Array{FT}(undef, Nx, Ny+1, Nz)
cm_cpu = Array{FT}(undef, Nx, Ny, Nz+1)
ps_cpu = Array{FT}(undef, Nx, Ny)
dam_cpu = Array{FT}(undef, Nx+1, Ny, Nz)
dbm_cpu = Array{FT}(undef, Nx, Ny+1, Nz)
dm_cpu  = Array{FT}(undef, Nx, Ny, Nz)
dcm_cpu = Array{FT}(undef, Nx, Ny, Nz+1)
close(r)

# Build grid for cluster sizes
config = AtmosTransport.IO.load_configuration("config/runs/era5_v4_diag_4win.toml")
model = AtmosTransport.IO.build_model_from_config(config)
cs_cpu = _get_cluster_sizes_cpu(model.grid)
lats = Array(model.grid.φᶜ_cpu)
lons = Array(model.grid.λᶜ_cpu)

# ── Create IC ────────────────────────────────────────────────────────
println("Creating structured IC (390-415 ppm vertical gradient)...")
# Load window 1 to get initial m
r1 = MassFluxBinaryReader(files[1], FT)
load_window!(m_cpu, am_cpu, bm_cpu, cm_cpu, ps_cpu, r1, 1)
close(r1)

rm_cpu = zeros(FT, Nx, Ny, Nz)
for k in 1:Nz
    vmr = FT(390e-6 + 25e-6 * (k - 1) / (Nz - 1))
    rm_cpu[:, :, k] .= vmr .* m_cpu[:, :, k]
end

# ── Create output NetCDF ─────────────────────────────────────────────
println("Creating output: $OUT_FILE")
isfile(OUT_FILE) && rm(OUT_FILE)
ds = NCDataset(OUT_FILE, "c")

defDim(ds, "lon", Nx)
defDim(ds, "lat", NJ_SUB)
defDim(ds, "lev", Nz)
defDim(ds, "lev_ifc", Nz+1)
defDim(ds, "lon_face", Nx+1)
defDim(ds, "lat_face", NJ_SUB+1)
defDim(ds, "sweep", N_SWEEPS)
defDim(ds, "substep", N_SUB)
defDim(ds, "window", N_WINDOWS)

# Coordinate variables
v_lon = defVar(ds, "lon", Float64, ("lon",))
v_lat = defVar(ds, "lat", Float64, ("lat",))
v_lon[:] = Float64.(lons)
v_lat[:] = Float64.(lats[J_RANGE])

# 3D fields per sweep×substep×window
v_rm = defVar(ds, "rm", FT, ("lon", "lat", "lev", "sweep", "substep", "window");
              attrib=Dict("long_name" => "tracer mass (CO2)", "units" => "kg"))
v_m  = defVar(ds, "m", FT, ("lon", "lat", "lev", "sweep", "substep", "window");
              attrib=Dict("long_name" => "air mass", "units" => "kg"))

# Fluxes (saved once per window, before advection)
v_am = defVar(ds, "am", FT, ("lon_face", "lat", "lev", "window");
              attrib=Dict("long_name" => "x mass flux", "units" => "kg"))
v_bm = defVar(ds, "bm", FT, ("lon", "lat_face", "lev", "window");
              attrib=Dict("long_name" => "y mass flux", "units" => "kg"))
v_cm = defVar(ds, "cm", FT, ("lon", "lat", "lev_ifc", "window");
              attrib=Dict("long_name" => "z mass flux", "units" => "kg"))

# ── Helper: save subregion ───────────────────────────────────────────
function save_fields!(ds, rm_gpu, m_gpu, sweep_idx, sub_idx, win_idx)
    rm_arr = Array(rm_gpu)
    m_arr  = Array(m_gpu)
    ds["rm"][:, :, :, sweep_idx, sub_idx, win_idx] = rm_arr[:, J_RANGE, :]
    ds["m"][:, :, :, sweep_idx, sub_idx, win_idx]  = m_arr[:, J_RANGE, :]
end

function save_fluxes!(ds, am_gpu, bm_gpu, cm_gpu, win_idx)
    am_arr = Array(am_gpu)
    bm_arr = Array(bm_gpu)
    cm_arr = Array(cm_gpu)
    # SH subregion: am has Nx+1 lons, bm has Ny+1 lats
    js = first(J_RANGE); je = last(J_RANGE)
    ds["am"][:, :, :, win_idx] = am_arr[:, J_RANGE, :]
    ds["bm"][:, :, :, win_idx] = bm_arr[:, js:je+1, :]  # +1 for face
    ds["cm"][:, :, :, win_idx] = cm_arr[:, J_RANGE, :]
end

# ── Run loop with per-sweep saving ───────────────────────────────────
println("\n=== Running 2-hour diagnostic with per-sweep output ===\n")

rm_gpu = CuArray(rm_cpu)
m_ref  = nothing  # will be set from each window's met data

for w in 1:N_WINDOWS
    println("── Window $w/$N_WINDOWS ──")

    # Load met data
    file_idx = (w - 1) ÷ 24 + 1
    local_w  = (w - 1) % 24 + 1
    r = MassFluxBinaryReader(files[file_idx], FT)
    load_window!(m_cpu, am_cpu, bm_cpu, cm_cpu, ps_cpu, r, local_w)
    if r.has_flux_delta
        load_flux_delta_window!(dam_cpu, dbm_cpu, dm_cpu, dcm_cpu, r, local_w)
    end
    close(r)

    # Apply preprocessing (boundaries + pole zeroing)
    cm_cpu[:,:,1] .= 0; cm_cpu[:,:,end] .= 0
    am_cpu[:, 1, :] .= 0; am_cpu[:, Ny, :] .= 0

    # Upload to GPU
    m_gpu  = CuArray(m_cpu)
    am_gpu = CuArray(am_cpu)
    bm_gpu = CuArray(bm_cpu)
    cm_gpu = CuArray(cm_cpu)

    # Save fluxes for this window
    save_fluxes!(ds, am_gpu, bm_gpu, cm_gpu, w)

    # Allocate workspace
    ws = allocate_massflux_workspace(m_gpu, am_gpu, bm_gpu, cm_gpu;
        cluster_sizes_cpu=cs_cpu)

    # Initialize m_dev from m_ref
    m_dev = copy(m_gpu)
    m_ref_gpu = copy(m_gpu)

    for s in 1:N_SUB
        # Single tracer
        rm_single = (co2=rm_gpu,)

        # Copy rm to workspace
        copyto!(ws.rm, rm_gpu)
        rm_ws = (co2=ws.rm,)

        # ── X1: forward X ──
        advect_x_massflux_subcycled!(rm_ws, m_dev, am_gpu, model.grid, true, ws; cfl_limit=FT(0.95))
        save_fields!(ds, ws.rm, m_dev, 1, s, w)

        # ── Y1: forward Y ──
        advect_y_massflux_subcycled!(rm_ws, m_dev, bm_gpu, model.grid, true, ws; cfl_limit=FT(0.95))
        save_fields!(ds, ws.rm, m_dev, 2, s, w)

        # ── Z1: forward Z ──
        advect_z_massflux_subcycled!(rm_ws, m_dev, cm_gpu, true, ws; cfl_limit=FT(0.95))
        save_fields!(ds, ws.rm, m_dev, 3, s, w)

        # ── Z2: reverse Z ──
        advect_z_massflux_subcycled!(rm_ws, m_dev, cm_gpu, true, ws; cfl_limit=FT(0.95))
        save_fields!(ds, ws.rm, m_dev, 4, s, w)

        # ── Y2: reverse Y ──
        advect_y_massflux_subcycled!(rm_ws, m_dev, bm_gpu, model.grid, true, ws; cfl_limit=FT(0.95))
        save_fields!(ds, ws.rm, m_dev, 5, s, w)

        # ── X2: reverse X ──
        advect_x_massflux_subcycled!(rm_ws, m_dev, am_gpu, model.grid, true, ws; cfl_limit=FT(0.95))
        save_fields!(ds, ws.rm, m_dev, 6, s, w)

        # Copy back to rm_gpu
        copyto!(rm_gpu, ws.rm)

        # Pole restore (constant-flux path)
        @views m_dev[:, 1, :]   .= m_ref_gpu[:, 1, :]
        @views m_dev[:, Ny, :] .= m_ref_gpu[:, Ny, :]

        # Print diagnostics
        rm_arr = Array(rm_gpu)
        m_arr  = Array(m_dev)
        q = rm_arr ./ max.(m_arr, FT(1))
        neg_m = sum(m_arr .< 0)
        nan_rm = sum(isnan, rm_arr)
        println("  sub $s: VMR $(round(minimum(q)*1e6,digits=1))-$(round(maximum(q)*1e6,digits=1))ppm, neg_m=$neg_m, NaN=$nan_rm")
    end

    # Update m_ref for next window
    copyto!(m_ref_gpu, m_dev)
end

close(ds)
println("\n✓ Saved to $OUT_FILE")
println("  Dimensions: lon=$Nx × lat=$NJ_SUB × lev=$Nz × sweep=$N_SWEEPS × substep=$N_SUB × window=$N_WINDOWS")

# Quick size estimate
fsize = filesize(OUT_FILE)
println("  File size: $(round(fsize/1e6, digits=1)) MB")
