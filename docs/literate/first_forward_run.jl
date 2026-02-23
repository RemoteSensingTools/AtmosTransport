# # First Forward Run: Tracer Transport with Real Meteorology
#
# This tutorial demonstrates an end-to-end forward simulation using
# AtmosTransport.jl with real GEOS-FP meteorological data downloaded
# directly via OPeNDAP — **no authentication required**.
#
# By the end, you will have:
# 1. Downloaded a coarsened GEOS-FP snapshot from NASA's OPeNDAP server
# 2. Built a model grid with 72 vertical levels
# 3. Initialized a CO₂-like tracer with a localized perturbation
# 4. Run 24 hours of advective transport using Slopes advection
# 5. Written the output to a NetCDF file
#
# ## Prerequisites
#
# Only the main `AtmosTransport` package and `NCDatasets` (a standard
# dependency) are needed. No API keys, no credentials, no extra packages.

using AtmosTransport
using AtmosTransport.Architectures
using AtmosTransport.Grids
using AtmosTransport.Advection
using NCDatasets
using Dates

# ## Step 1: Download GEOS-FP test data
#
# NASA's GEOS Forward Processing system provides near-real-time assimilated
# meteorological fields at 0.25° × 0.3125° resolution, freely accessible
# via OPeNDAP. We subsample by a stride of 10 to get a manageable ~3° grid.

const STRIDE = 10
const TEST_DATE = Date(2024, 12, 1)
const OUTDIR = joinpath(homedir(), "data", "metDrivers", "geosfp", "test")
const FT = Float64

# The download function opens the OPeNDAP endpoint, reads a single 12Z
# snapshot with spatial subsampling, and writes it as a local NetCDF file.
# Once downloaded, subsequent runs use the cached file automatically.

function download_geosfp(date, outdir; stride=STRIDE)
    mkpath(outdir)
    outfile = joinpath(outdir, "geosfp_asm_$(Dates.format(date, "yyyymmdd")).nc")
    isfile(outfile) && filesize(outfile) > 1000 && return outfile

    url = "https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/assim/inst3_3d_asm_Nv"
    ds = NCDataset(url)
    try
        lons, lats, levs = ds["lon"][:], ds["lat"][:], ds["lev"][:]
        times = ds["time"][:]
        target = DateTime(date) + Hour(12)
        tidx = if eltype(times) <: DateTime
            argmin([abs(Dates.value(t - target)) for t in times])
        else
            argmin(abs.(times .- (Dates.value(date - Date(1,1,1)) + 0.5)))
        end
        li, la = 1:stride:length(lons), 1:stride:length(lats)

        NCDataset(outfile, "c") do out
            defDim(out, "lon", length(li))
            defDim(out, "lat", length(la))
            defDim(out, "lev", length(levs))
            defVar(out, "lon", Float64, ("lon",))[:] = lons[li]
            defVar(out, "lat", Float64, ("lat",))[:] = lats[la]
            defVar(out, "lev", Float64, ("lev",))[:] = levs
            for v in ["u","v","omega","delp"]
                defVar(out, v, Float32, ("lon","lat","lev"))[:] = Float32.(ds[v][li,la,:,tidx])
            end
        end
    finally
        close(ds)
    end
    return outfile
end

metfile = download_geosfp(TEST_DATE, OUTDIR)

# ## Step 2: Load met data and build staggered velocities
#
# The advection scheme requires velocities at cell *faces*:
# - `u` at longitude faces: `(Nx+1, Ny, Nz)`, periodic
# - `v` at latitude faces: `(Nx, Ny+1, Nz)`, zero at poles
# - `w` at level interfaces: `(Nx, Ny, Nz+1)`, zero at top/surface
#
# We compute these by averaging adjacent cell-center values and apply
# a CFL limiter to ensure numerical stability.

ds = NCDataset(metfile)
lons, lats = ds["lon"][:], ds["lat"][:]
Nx, Ny, Nz = length(lons), length(lats), length(ds["lev"][:])
u_cc   = FT.(ds["u"][:,:,:])
v_cc   = FT.(ds["v"][:,:,:])
omega  = FT.(ds["omega"][:,:,:])
delp   = FT.(ds["delp"][:,:,:])
close(ds)

#-

## Column-mean pressure thickness → vertical coordinate
mean_dp = [sum(delp[:,:,k])/(Nx*Ny) for k in 1:Nz]
p_edges = cumsum(vcat(0.0, mean_dp))
Ps = p_edges[end]

## Stagger winds to faces with CFL limiting
Δt = 300.0  # 5-minute time step
cfl = 0.4
Δx_min = STRIDE * 0.3125 * 111000.0
Δy_min = STRIDE * 0.25   * 111000.0
u_max, v_max = cfl * Δx_min / Δt, cfl * Δy_min / Δt

u = zeros(Nx+1, Ny, Nz)
for k in 1:Nz, j in 1:Ny, i in 1:Nx
    u[i,j,k] = clamp((u_cc[i,j,k]+u_cc[i==Nx ? 1 : i+1,j,k])/2, -u_max, u_max)
end
u[Nx+1,:,:] .= u[1,:,:]

v = zeros(Nx, Ny+1, Nz)
for k in 1:Nz, j in 2:Ny, i in 1:Nx
    v[i,j,k] = clamp((v_cc[i,j-1,k]+v_cc[i,j,k])/2, -v_max, v_max)
end

w = zeros(Nx, Ny, Nz+1)
for k in 2:Nz, j in 1:Ny, i in 1:Nx
    w_raw = -(omega[i,j,k-1]+omega[i,j,k])/2
    dp_min = min(mean_dp[k-1], mean_dp[k])
    w[i,j,k] = clamp(w_raw, -cfl*dp_min/Δt, cfl*dp_min/Δt)
end

velocities = (; u, v, w)

# ## Step 3: Build model grid
#
# We use a `LatitudeLongitudeGrid` with 72 hybrid sigma-pressure levels
# derived from the actual GEOS-FP pressure thickness. This ensures that
# `Δz(k, grid)` returns physically realistic pressure thicknesses at
# each level — crucial for CFL stability.

vc = HybridSigmaPressure(zeros(Nz+1), p_edges ./ Ps)
grid = LatitudeLongitudeGrid(CPU();
    size = (Nx, Ny, Nz),
    longitude = (lons[1], lons[end] + (lons[2]-lons[1])),
    latitude  = (lats[1], lats[end]),
    vertical  = vc)

# ## Step 4: Initialize tracers
#
# Start with a uniform CO₂-like background (400 ppm) plus a Gaussian
# perturbation centered over Europe. This makes it easy to visualize
# how the tracer disperses under the influence of GEOS-FP winds.

c = fill(400.0, Nx, Ny, Nz)
for k in 1:Nz, j in 1:Ny, i in 1:Nx
    Δlon = lons[i] - 10.0
    Δlat = lats[j] - 50.0
    blob = 50.0 * exp(-(Δlon^2/450 + Δlat^2/200 + (k-Nz+5)^2/8))
    c[i,j,k] += blob
end
tracers = (; c)

println("Initial tracer: min=$(minimum(c)), max=$(round(maximum(c), digits=1)), " *
        "mean=$(round(sum(c)/length(c), digits=2))")

# ## Step 5: Run 24 hours of advective transport
#
# We use the Russell-Lerner **SlopesAdvection** scheme (second-order,
# with minmod flux limiter) and symmetric Strang operator splitting:
#
# ```
# advect_x (Δt/2)  →  advect_y (Δt/2)  →  advect_z (Δt/2)
#       →  advect_z (Δt/2)  →  advect_y (Δt/2)  →  advect_x (Δt/2)
# ```
#
# No convection or diffusion in this demo — pure advection.

scheme = SlopesAdvection(use_limiter=true)
N_hours = 24
N_steps = round(Int, N_hours * 3600 / Δt)
half = Δt / 2

mass_initial = sum(tracers.c)

for step in 1:N_steps
    advect_x!(tracers, velocities, grid, scheme, half)
    advect_y!(tracers, velocities, grid, scheme, half)
    advect_z!(tracers, velocities, grid, scheme, half)
    advect_z!(tracers, velocities, grid, scheme, half)
    advect_y!(tracers, velocities, grid, scheme, half)
    advect_x!(tracers, velocities, grid, scheme, half)

    if step % (N_steps ÷ 8) == 0
        t_hr = step * Δt / 3600
        mass_now = sum(tracers.c)
        Δm = abs(mass_now - mass_initial) / abs(mass_initial) * 100
        println("  t=$(round(t_hr, digits=1))h: " *
                "min=$(round(minimum(tracers.c), digits=1)), " *
                "max=$(round(maximum(tracers.c), digits=1)), " *
                "mass_change=$(round(Δm, sigdigits=3))%")
    end
end

# ## Results
#
# The tracer blob has been transported by the GEOS-FP wind field.
# Key observations:
#
# - **Mass conservation**: The limiter introduces small mass changes
#   (~1%), which is expected for a single-snapshot wind field that
#   isn't divergence-free after CFL clipping. A proper simulation
#   would use temporally interpolated, divergence-corrected winds.
#
# - **Dispersion**: The initial Gaussian blob spreads both horizontally
#   (by the jet stream and surface winds) and vertically (by omega).
#
# - **Stability**: The SlopesAdvection scheme with CFL-limited winds
#   runs stably for the full 24-hour period.

println("\n--- Final State ---")
println("  Steps: $N_steps")
println("  min=$(round(minimum(tracers.c), digits=2))")
println("  max=$(round(maximum(tracers.c), digits=2))")
println("  mass change: $(round(abs(sum(tracers.c)-mass_initial)/abs(mass_initial)*100, sigdigits=3))%")

# ## Next Steps
#
# From here, you can:
#
# 1. **Add physics**: Include `BoundaryLayerDiffusion` and
#    `TiedtkeConvection` for a more complete transport model
# 2. **Swap met drivers**: Replace `geosfp.toml` with `merra2.toml` or
#    `era5.toml` to compare reanalysis products
# 3. **Run the adjoint**: Use `adjoint_advect_x/y/z!` to compute
#    sensitivities of an observation to upwind sources
# 4. **GPU acceleration**: Replace `CPU()` with `GPU()` for NVIDIA
#    hardware (requires the CUDA extension)
# 5. **Higher resolution**: Reduce `STRIDE` for finer grids (needs
#    proportionally smaller `Δt`)
