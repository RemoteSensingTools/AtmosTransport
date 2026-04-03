#!/usr/bin/env julia
# Check whether spectral mass fluxes satisfy the continuity equation exactly.
# If Σ(flux divergence) ≈ 0 globally, the mass fluxes are consistent.
# Any nonzero residual means the preprocessor or the advection scheme leaks mass.
#
# Usage:
#   julia --project=. scripts/diagnostics/check_mass_flux_continuity.jl \
#       ~/data/AtmosTransport/met/era5/preprocessed_spectral_catrine/massflux_era5_spectral_202112_float32.nc

using NCDatasets, Printf

path = length(ARGS) >= 1 ? expanduser(ARGS[1]) :
    expanduser("~/data/AtmosTransport/met/era5/preprocessed_spectral_catrine/massflux_era5_spectral_202112_float32.nc")

ds = NCDataset(path, "r")

Nt = ds.dim["time"]
println("File: $path")
println("Dimensions: lon=$(ds.dim["lon"]), lat=$(ds.dim["lat"]), lev=$(ds.dim["lev"]), time=$Nt")
println()

# Check first few windows
for t in 1:min(Nt, 5)
    # Read in Float64 for precision
    m_t  = Float64.(ds["m"][:, :, :, t])    # (lon, lat, lev)
    am_t = Float64.(ds["am"][:, :, :, t])   # (lon_u, lat, lev)
    bm_t = Float64.(ds["bm"][:, :, :, t])   # (lon, lat_v, lev)
    cm_t = Float64.(ds["cm"][:, :, :, t])   # (lon, lat, lev_w)

    Nx, Ny, Nz = size(m_t)

    # Global flux divergence: should telescope to boundary fluxes only
    # X: periodic → Σ(am[i] - am[i+1]) over all i = 0
    # Y: closed poles → Σ(bm[j] - bm[j+1]) over all j = bm[1] - bm[Ny+1]
    # Z: closed top/bottom → Σ(cm[k] - cm[k+1]) over all k = cm[1] - cm[Nz+1]

    # Per-cell divergence check
    max_res = 0.0
    sum_div = 0.0
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        div_x = am_t[i, j, k] - am_t[i + 1, j, k]
        div_y = bm_t[i, j, k] - bm_t[i, j + 1, k]
        div_z = cm_t[i, j, k] - cm_t[i, j, k + 1]
        div_total = div_x + div_y + div_z
        sum_div += div_total
        max_res = max(max_res, abs(div_total))
    end

    # Also check mass tendency if next window exists
    if t < Nt
        m_next = Float64.(ds["m"][:, :, :, t + 1])
        dm = sum(m_next) - sum(m_t)
        @printf("Window %2d: Σdiv = %+.6e  |  Σm = %.6e  |  Δm(t→t+1) = %+.6e  |  max|div| = %.3e\n",
                t, sum_div, sum(m_t), dm, max_res)
    else
        @printf("Window %2d: Σdiv = %+.6e  |  Σm = %.6e  |  max|div| = %.3e\n",
                t, sum_div, sum(m_t), max_res)
    end
end

close(ds)
println("\nIf Σdiv ≈ 0 and max|div| is small, mass fluxes are consistent.")
println("If Σdiv ≠ 0, the preprocessor has a mass flux imbalance.")
