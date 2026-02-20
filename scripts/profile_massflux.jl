#!/usr/bin/env julia
# Quick profiling to identify mass-flux advection bottleneck on GPU

using AtmosTransportModel
using AtmosTransportModel.Architectures
using AtmosTransportModel.Grids
using AtmosTransportModel.Advection
using AtmosTransportModel.Parameters
using AtmosTransportModel.IO: default_met_config, build_vertical_coordinate,
                              load_vertical_coefficients
using NCDatasets
using KernelAbstractions: synchronize, get_backend

const USE_GPU = parse(Bool, get(ENV, "USE_GPU", "false"))
if USE_GPU
    using CUDA
    CUDA.allowscalar(false)
end

const FT = Float64
const LEVEL_TOP = 50
const LEVEL_BOT = 137
const LEVEL_RANGE = LEVEL_TOP:LEVEL_BOT
const DATADIR = expanduser("~/data/metDrivers/era5/era5_ml_10deg_20240601_20240607")

function load_era5_timestep(filepath, tidx, ::Type{FT}) where FT
    ds = NCDataset(filepath)
    try
        u = FT.(ds["u"][:, :, :, tidx])[:, end:-1:1, :]
        v = FT.(ds["v"][:, :, :, tidx])[:, end:-1:1, :]
        lnsp = FT.(ds["lnsp"][:, :, tidx])[:, end:-1:1]
        return u, v, exp.(lnsp)
    finally
        close(ds)
    end
end

function stagger_u_v(u_cc, v_cc, Nx, Ny, Nz, ::Type{FT}) where FT
    u = zeros(FT, Nx + 1, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        ip = i == Nx ? 1 : i + 1
        u[i, j, k] = (u_cc[i, j, k] + u_cc[ip, j, k]) / 2
    end
    u[Nx + 1, :, :] .= u[1, :, :]
    v = zeros(FT, Nx, Ny + 1, Nz)
    @inbounds for k in 1:Nz, j in 2:Ny, i in 1:Nx
        v[i, j, k] = (v_cc[i, j - 1, k] + v_cc[i, j, k]) / 2
    end
    return u, v
end

# -- Load one timestep --
file1 = joinpath(DATADIR, "era5_ml_20240601.nc")
ds = NCDataset(file1)
lons = FT.(ds["longitude"][:])
lats = FT.(reverse(ds["latitude"][:]))
close(ds)
Nx, Ny = length(lons), length(lats)
Nz = length(LEVEL_RANGE)

config = default_met_config("era5")
vc = build_vertical_coordinate(config; FT, level_range=LEVEL_RANGE)
params = load_parameters(FT)
pp = params.planet

arch = USE_GPU ? GPU() : CPU()
AT = array_type(arch)
Δlon = lons[2] - lons[1]

grid = LatitudeLongitudeGrid(arch;
    FT, size = (Nx, Ny, Nz),
    longitude = (FT(lons[1]), FT(lons[end]) + FT(Δlon)),
    latitude = (FT(-90), FT(90)),
    vertical = vc,
    radius = pp.radius, gravity = pp.gravity,
    reference_pressure = pp.reference_surface_pressure)

u_cc, v_cc, ps = load_era5_timestep(file1, 1, FT)
u_stag, v_stag = stagger_u_v(u_cc, v_cc, Nx, Ny, Nz, FT)

Δp_cpu = Advection._build_Δz_3d(grid, ps)
Δp = AT(Δp_cpu)
u_dev = AT(u_stag)
v_dev = AT(v_stag)

@info "Grid: $Nx × $Ny × $Nz, arch=$arch"

# -- Compute mass & fluxes --
t0 = time()
m = compute_air_mass(Δp, grid)
@info "compute_air_mass: $(round(time()-t0, digits=3))s"

t0 = time()
mf = compute_mass_fluxes(u_dev, v_dev, grid, Δp, FT(450))
@info "compute_mass_fluxes: $(round(time()-t0, digits=3))s"

# -- CFL diagnostics --
t0 = time()
cfl_x = max_cfl_massflux_x(mf.am, m)
@info "CFL_x = $(round(cfl_x, digits=2)) ($(round(time()-t0, digits=3))s)"

t0 = time()
cfl_y = max_cfl_massflux_y(mf.bm, m)
@info "CFL_y = $(round(cfl_y, digits=2)) ($(round(time()-t0, digits=3))s)"

t0 = time()
cfl_z = max_cfl_massflux_z(mf.cm, m)
@info "CFL_z = $(round(cfl_z, digits=2)) ($(round(time()-t0, digits=3))s)"

n_sub_x = max(1, ceil(Int, cfl_x / 0.95))
n_sub_y = max(1, ceil(Int, cfl_y / 0.95))
n_sub_z = max(1, ceil(Int, cfl_z / 0.95))
@info "Subcycles: x=$n_sub_x, y=$n_sub_y, z=$n_sub_z"

# -- Allocate workspace --
ws = allocate_massflux_workspace(m, mf.am, mf.bm, mf.cm)

# -- Initialize tracer --
c = AT(fill(FT(420.0), Nx, Ny, Nz))
tracers = (; c)

# -- Warmup: run one Strang split --
@info "\n--- Warmup (includes kernel compilation) ---"
t0 = time()
strang_split_massflux!(tracers, m, mf.am, mf.bm, mf.cm, grid, true, ws;
                        cfl_limit = FT(0.95))
t1 = time()
@info "  First Strang split: $(round(t1-t0, digits=2))s"

# -- Benchmark: 10 Strang splits --
@info "\n--- Benchmark: 10 Strang splits ---"
t0 = time()
for i in 1:10
    strang_split_massflux!(tracers, m, mf.am, mf.bm, mf.cm, grid, true, ws;
                            cfl_limit = FT(0.95))
end
t1 = time()
avg = round((t1-t0)/10, digits=4)
@info "  10 splits in $(round(t1-t0, digits=2))s, avg=$(avg)s/split"

c_host = Array(tracers.c)
@info "  min=$(round(minimum(c_host), digits=4)), max=$(round(maximum(c_host), digits=4))"
@info "  mean=$(round(sum(c_host)/length(c_host), digits=4))"
