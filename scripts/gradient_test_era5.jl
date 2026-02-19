#!/usr/bin/env julia
# ===========================================================================
# Gradient test on ERA5 winds
#
# Tests the adjoint correctness of the full operator-splitting time stepper
# with realistic ERA5 wind fields (not constant 5 m/s).
#
# Runs three configurations:
#   1. SlopesAdvection(use_limiter=false) — expect ratio = 1.0 (exact discrete adjoint)
#   2. SlopesAdvection(use_limiter=true)  — expect ratio ~ 1.0 (continuous adjoint)
#   3. UpwindAdvection()                  — expect ratio = 1.0 (exact discrete adjoint)
#
# Usage:
#   julia --project=. scripts/gradient_test_era5.jl
# ===========================================================================

using AtmosTransportModel
using AtmosTransportModel.Architectures
using AtmosTransportModel.Grids
using AtmosTransportModel.Advection
using AtmosTransportModel.Convection
using AtmosTransportModel.Diffusion
using AtmosTransportModel.Chemistry
using AtmosTransportModel.TimeSteppers
using AtmosTransportModel.Adjoint
using AtmosTransportModel.Parameters
using NCDatasets
using LinearAlgebra: dot
using Random

const FT = Float64
const OVERRIDE_TOML = get(ENV, "CONFIG", nothing)
const PARAMS = load_parameters(FT; override=OVERRIDE_TOML)

const PROJECT_ROOT = abspath(joinpath(@__DIR__, ".."))
const PL_FILE = joinpath(PROJECT_ROOT, "data", "era5", "era5_pressure_levels_20250201_20250207.nc")
const SL_FILE = joinpath(PROJECT_ROOT, "data", "era5", "era5_single_levels_20250201_20250207.nc")
const Δt = FT(600.0)

@info "=" ^ 60
@info "Gradient Test on ERA5 Winds"
@info "=" ^ 60

# ---------------------------------------------------------------------------
# Load ERA5 data and build grid (subset: smaller grid for speed)
# ---------------------------------------------------------------------------
@info "\n--- Loading ERA5 data ---"

ds = NCDataset(PL_FILE)
lon_key = haskey(ds, "longitude") ? "longitude" : "lon"
lat_key = haskey(ds, "latitude") ? "latitude" : "lat"
lev_key = haskey(ds, "pressure_level") ? "pressure_level" : "level"

lons_raw = FT.(ds[lon_key][:])
lats_raw = FT.(ds[lat_key][:])
levels_raw = ds[lev_key][:]

u_raw = FT.(ds["u"][:, :, :, 1])
v_raw = FT.(ds["v"][:, :, :, 1])
close(ds)

Nx_full, Ny_full, Nz_full = length(lons_raw), length(lats_raw), length(levels_raw)

# Shift/reverse as in run_forward_era5.jl
if lons_raw[1] >= 0 && lons_raw[end] > 180
    shift_idx = findfirst(l -> l >= 180, lons_raw)
    shift_n = Nx_full - shift_idx + 1
    lons_raw = circshift(lons_raw, shift_n)
    lons_raw[1:shift_n] .-= FT(360)
    u_raw = circshift(u_raw, (shift_n, 0, 0))
    v_raw = circshift(v_raw, (shift_n, 0, 0))
end
if lats_raw[1] > lats_raw[end]
    lats_raw = reverse(lats_raw)
    u_raw = u_raw[:, end:-1:1, :]
    v_raw = v_raw[:, end:-1:1, :]
end
if levels_raw[1] > levels_raw[end]
    levels_raw = reverse(levels_raw)
    u_raw = u_raw[:, :, end:-1:1]
    v_raw = v_raw[:, :, end:-1:1]
end

Nx, Ny, Nz = Nx_full, Ny_full, Nz_full
@info "  Grid: $Nx × $Ny × $Nz"

# Build grid
p_levels_Pa = FT.(levels_raw) .* FT(100)
p_edges = zeros(FT, Nz + 1)
p_edges[1] = FT(0)
for k in 1:Nz-1
    p_edges[k+1] = (p_levels_Pa[k] + p_levels_Pa[k+1]) / 2
end
p_edges[Nz+1] = PARAMS.planet.reference_surface_pressure

Δlon = lons_raw[2] - lons_raw[1]
Ps = p_edges[end]
b_vals = p_edges ./ Ps
a_vals = zeros(FT, Nz + 1)
vc = HybridSigmaPressure(a_vals, b_vals)

pp = PARAMS.planet
grid = LatitudeLongitudeGrid(CPU(); FT,
    size = (Nx, Ny, Nz),
    longitude = (lons_raw[1] - Δlon/2, lons_raw[end] + Δlon/2),
    latitude = (FT(-90), FT(90)),
    vertical = vc,
    radius = pp.radius,
    gravity = pp.gravity,
    reference_pressure = pp.reference_surface_pressure)

# Stagger u, v and diagnose w
@info "  Staggering winds..."
φᶠ_cpu = Array(grid.φᶠ)
R_earth = grid.radius

u = zeros(FT, Nx + 1, Ny, Nz)
@inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
    ip = i == Nx ? 1 : i + 1
    u[i, j, k] = (u_raw[i, j, k] + u_raw[ip, j, k]) / 2
end
u[Nx + 1, :, :] .= u[1, :, :]

v = zeros(FT, Nx, Ny + 1, Nz)
@inbounds for k in 1:Nz, j in 2:Ny, i in 1:Nx
    v[i, j, k] = (v_raw[i, j-1, k] + v_raw[i, j, k]) / 2
end

div_h = zeros(FT, Nx, Ny, Nz)
@inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
    ip = i == Nx ? 1 : i + 1
    dx = Δx(i, j, grid)
    div_u = (u[ip, j, k] - u[i, j, k]) / dx
    cos_N = cosd(φᶠ_cpu[j + 1])
    cos_S = cosd(φᶠ_cpu[j])
    sin_N = sind(φᶠ_cpu[j + 1])
    sin_S = sind(φᶠ_cpu[j])
    dsinphi = max(abs(sin_N - sin_S), FT(1e-30))
    div_v = (v[i, j+1, k] * cos_N - v[i, j, k] * cos_S) / (R_earth * dsinphi)
    div_h[i, j, k] = div_u + div_v
end

P_total = p_edges[Nz+1] - p_edges[1]
@inbounds for j in 1:Ny, i in 1:Nx
    pit = FT(0)
    for k in 1:Nz
        pit += div_h[i, j, k] * (p_edges[k+1] - p_edges[k])
    end
    for k in 1:Nz
        div_h[i, j, k] -= pit / P_total
    end
end

w = zeros(FT, Nx, Ny, Nz + 1)
@inbounds for j in 1:Ny, i in 1:Nx
    for k in 1:Nz
        w[i, j, k+1] = w[i, j, k] - div_h[i, j, k] * (p_edges[k+1] - p_edges[k])
    end
end

met = (; u, v, w)
@info "  Wind ranges: u=[$(round(minimum(u),digits=1)),$(round(maximum(u),digits=1))] m/s"
@info "               v=[$(round(minimum(v),digits=1)),$(round(maximum(v),digits=1))] m/s"
@info "               w=[$(round(minimum(w),digits=3)),$(round(maximum(w),digits=3))] Pa/s"

# ---------------------------------------------------------------------------
# Gradient test
# ---------------------------------------------------------------------------
N_STEPS = 3

function run_gradient_test(scheme_name, scheme; n_steps=N_STEPS, verbose=true)
    @info "\n--- $scheme_name ---"

    ts = OperatorSplittingTimeStepper(
        advection  = scheme,
        convection = NoConvection(),
        diffusion  = NoDiffusion(),
        chemistry  = NoChemistry(),
        Δt_outer   = Δt)

    results = gradient_test(; grid, timestepper=ts, met_data=met,
                              n_steps=n_steps, Δt=Δt,
                              epsilons=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
                              verbose=verbose)

    @info "  Summary:"
    for (ε, ratio) in results
        status = abs(ratio - 1.0) < 0.01 ? "OK" : (abs(ratio - 1.0) < 0.5 ? "~OK" : "FAIL")
        @info "    ε=$ε  ratio=$(round(ratio, sigdigits=10))  [$status]"
    end

    return results
end

# 1. SlopesAdvection without limiter (exact discrete adjoint)
results_slopes_nolim = run_gradient_test(
    "SlopesAdvection (no limiter) — exact discrete adjoint",
    SlopesAdvection(use_limiter=false))

# 2. SlopesAdvection with limiter (continuous adjoint)
results_slopes_lim = run_gradient_test(
    "SlopesAdvection (with limiter) — continuous adjoint",
    SlopesAdvection(use_limiter=true))

# 3. UpwindAdvection (exact discrete adjoint)
results_upwind = run_gradient_test(
    "UpwindAdvection — exact discrete adjoint",
    UpwindAdvection())

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
@info "\n" * "=" ^ 60
@info "SUMMARY"
@info "=" ^ 60

function report(name, results)
    best_ε = results[end]
    mid_ε = results[3]
    @info "  $name:"
    @info "    Best ratio (ε=$(best_ε[1])): $(round(best_ε[2], sigdigits=10))"
    @info "    Mid  ratio (ε=$(mid_ε[1])): $(round(mid_ε[2], sigdigits=10))"
end

report("Slopes (no limiter)", results_slopes_nolim)
report("Slopes (with limiter)", results_slopes_lim)
report("Upwind", results_upwind)

@info "\n" * "=" ^ 60
