#!/usr/bin/env julia
# Visualize the per-cell mass balance closure residual from preprocessed binary.
#
# R[i,j,k] = bt[k] × pit[i,j] - dm[i,j,k]
#
# This is what the mass fixer must correct each window. Large R at fronts
# explains the Rossby-wave noise pattern in the transport.

using Pkg; Pkg.activate(".")
using AtmosTransport
using AtmosTransport.IO: MassFluxBinaryReader, load_window!, LatLonCPUBuffer
using Statistics, Printf
using NCDatasets

# --- Config ---
data_dir = "/temp1/atmos_transport/era5_v4_5000Pa_balanced"
outfile = "/tmp/mass_residual_diag.nc"

files = sort(filter(f -> endswith(f, ".bin"), readdir(data_dir; join=true)))
@info "Found $(length(files)) binary files"

rdr1 = MassFluxBinaryReader(files[1], Float32)
Nx, Ny, Nz = rdr1.Nx, rdr1.Ny, rdr1.Nz
n_win_file = rdr1.Nt
@info "Grid: $Nx×$Ny×$Nz, $n_win_file windows/file"

# B-correction vector
B_ifc = rdr1.B_ifc
A_ifc = rdr1.A_ifc
dB = Float32[Float32(B_ifc[k+1] - B_ifc[k]) for k in 1:Nz]
dB_total = Float32(B_ifc[Nz+1] - B_ifc[1])
bt = abs(dB_total) > eps(Float32) ? dB ./ dB_total : zeros(Float32, Nz)
@info "bt: sum=$(sum(bt)), range $(extrema(bt))"

FT = Float32

# Allocate raw arrays for load_window!
_m  = zeros(FT, Nx, Ny, Nz)
_am = zeros(FT, Nx+1, Ny, Nz)
_bm = zeros(FT, Nx, Ny+1, Nz)
_cm = zeros(FT, Nx, Ny, Nz+1)
_ps = zeros(FT, Nx, Ny)

# Storage for per-window snapshots (save first 48 windows as maps)
n_windows = min(47, length(files) * n_win_file - 1)  # need w+1

# Output: per-window surface + 750hPa residual maps, and per-level zonal stats
residual_sfc_all  = zeros(FT, Nx, Ny, n_windows)
residual_750_all  = zeros(FT, Nx, Ny, n_windows)
residual_col_all  = zeros(FT, Nx, Ny, n_windows)  # column-integrated |R|
dm_sfc_all        = zeros(FT, Nx, Ny, n_windows)
rel_residual_sfc  = zeros(FT, Nx, Ny, n_windows)

# Find ~750 hPa level (bt peaks near surface; for 18L merged, level ~14)
k750 = max(1, round(Int, 0.74 * Nz))
@info "Using k=$k750 for ~750 hPa slice"

function load_m_am_bm(files, w)
    file_idx = (w - 1) ÷ n_win_file + 1
    win_in_file = (w - 1) % n_win_file + 1
    rdr = MassFluxBinaryReader(files[file_idx], Float32)
    load_window!(_m, _am, _bm, _cm, _ps, rdr, win_in_file)
    return copy(_m), copy(_am), copy(_bm)
end

for w in 1:n_windows
    m_curr, am, bm = load_m_am_bm(files, w)
    m_next, _, _ = load_m_am_bm(files, w + 1)

    dm = m_next .- m_curr

    # Horizontal divergence per cell (convergence is positive → mass gain)
    div_h = zeros(FT, Nx, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        # am is (Nx+1, Ny, Nz) with periodic x: am[Nx+1,j,k] = am[1,j,k]
        am_out = i < Nx ? am[i+1, j, k] : am[1, j, k]
        div_h[i, j, k] = (am[i, j, k] - am_out) + (bm[i, j, k] - bm[i, j+1, k])
    end

    # Column-integrated divergence
    pit = dropdims(sum(div_h, dims=3), dims=3)

    # B-correction implied per-level mass change
    # R[k] = bt[k] × pit - dm[k]
    R = zeros(FT, Nx, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        R[i, j, k] = bt[k] * pit[i, j] - dm[i, j, k]
    end

    # Store snapshots
    residual_sfc_all[:, :, w] = R[:, :, Nz]
    residual_750_all[:, :, w] = R[:, :, k750]
    residual_col_all[:, :, w] = dropdims(sum(abs.(R), dims=3), dims=3)
    dm_sfc_all[:, :, w] = dm[:, :, Nz]
    safe_dm = max.(abs.(dm[:, :, Nz]), FT(1))
    rel_residual_sfc[:, :, w] = R[:, :, Nz] ./ safe_dm

    if w <= 5 || w % 10 == 0
        @printf("Win %3d: R_rms_sfc=%.3e R_rms_750=%.3e dm_rms=%.3e ratio=%.4f\n",
                w, sqrt(mean(R[:,:,Nz].^2)), sqrt(mean(R[:,:,k750].^2)),
                sqrt(mean(dm[:,:,Nz].^2)),
                sqrt(mean(R[:,:,Nz].^2)) / max(sqrt(mean(dm[:,:,Nz].^2)), 1e-30))
    end
end

# --- Save to NetCDF ---
lons = range(-179.75f0, 179.75f0, length=Nx)
lats = range(-89.75f0, 89.75f0, length=Ny)

rm(outfile, force=true)
ds = NCDataset(outfile, "c")
defDim(ds, "lon", Nx)
defDim(ds, "lat", Ny)
defDim(ds, "lev", Nz)
defDim(ds, "time", n_windows)

defVar(ds, "lon", Float32, ("lon",))[:] = collect(lons)
defVar(ds, "lat", Float32, ("lat",))[:] = collect(lats)
defVar(ds, "bt", Float32, ("lev",))[:] = bt

defVar(ds, "R_sfc", Float32, ("lon", "lat", "time"))[:] = residual_sfc_all
defVar(ds, "R_750", Float32, ("lon", "lat", "time"))[:] = residual_750_all
defVar(ds, "R_col", Float32, ("lon", "lat", "time"))[:] = residual_col_all
defVar(ds, "dm_sfc", Float32, ("lon", "lat", "time"))[:] = dm_sfc_all
defVar(ds, "rel_R_sfc", Float32, ("lon", "lat", "time"))[:] = rel_residual_sfc

close(ds)
@info "Saved: $outfile"
