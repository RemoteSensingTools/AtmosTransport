#!/usr/bin/env julia
# Minimal GPU kernel timing: load 1 window, run N strang splits, measure wall time

using AtmosTransport
using AtmosTransport.Architectures
using AtmosTransport.Grids
using AtmosTransport.Advection
using AtmosTransport.Parameters
using AtmosTransport.IO: default_met_config, build_vertical_coordinate
using NCDatasets
using Printf
using CUDA
CUDA.allowscalar(false)

const FT = Float32
const MASSFLUX_FILE = expanduser(get(ENV, "MASSFLUX_FILE",
    "~/data/metDrivers/era5/massflux_era5_202406_float32.nc"))

function profile_gpu()
    @info "=" ^ 70
    @info "GPU Kernel Timing Profile"
    @info "=" ^ 70

    ds = NCDataset(MASSFLUX_FILE)
    lons = FT.(ds["lon"][:])
    lats = FT.(ds["lat"][:])
    Nx, Ny = length(lons), length(lats)
    Nz = ds.dim["lev"]
    level_top = ds.attrib["level_top"]
    level_bot = ds.attrib["level_bot"]

    config = default_met_config("era5")
    vc = build_vertical_coordinate(config; FT, level_range=level_top:level_bot)
    params = load_parameters(FT)
    pp = params.planet

    Δlon = lons[2] - lons[1]
    grid = LatitudeLongitudeGrid(GPU();
        FT, size = (Nx, Ny, Nz),
        longitude = (FT(lons[1]), FT(lons[end]) + FT(Δlon)),
        latitude = (FT(-90), FT(90)),
        vertical = vc,
        radius = pp.radius, gravity = pp.gravity,
        reference_pressure = pp.reference_surface_pressure)

    m_cpu  = FT.(ds["m"][:, :, :, 1])
    am_cpu = FT.(ds["am"][:, :, :, 1])
    bm_cpu = FT.(ds["bm"][:, :, :, 1])
    cm_cpu = FT.(ds["cm"][:, :, :, 1])
    close(ds)

    m_dev  = CuArray(m_cpu)
    am_dev = CuArray(am_cpu)
    bm_dev = CuArray(bm_cpu)
    cm_dev = CuArray(cm_cpu)
    c      = CUDA.zeros(FT, Nx, Ny, Nz)
    c .+= FT(1e-3)
    tracers = (; c)
    ws = allocate_massflux_workspace(m_dev, am_dev, bm_dev, cm_dev)

    @info "Grid: $Nx × $Ny × $Nz = $(Nx*Ny*Nz) cells"
    @info "GPU: $(CUDA.name(CUDA.device()))"
    @info ""

    @info "--- Phase 1: Kernel compilation (first call) ---"
    t0 = time()
    strang_split_massflux!(tracers, m_dev, am_dev, bm_dev, cm_dev,
                           grid, true, ws; cfl_limit = FT(0.95))
    CUDA.synchronize()
    t_compile = time() - t0
    @info @sprintf("  First strang_split: %.2fs (includes CUDA compilation)", t_compile)

    @info "\n--- Phase 2: Warm runs ---"
    for trial in 1:3
        m_dev2 = CuArray(m_cpu)
        c2 = CUDA.zeros(FT, Nx, Ny, Nz) .+ FT(1e-3)
        tracers2 = (; c = c2)
        ws2 = allocate_massflux_workspace(m_dev2, am_dev, bm_dev, cm_dev)

        CUDA.synchronize()
        t0 = time()
        strang_split_massflux!(tracers2, m_dev2, am_dev, bm_dev, cm_dev,
                               grid, true, ws2; cfl_limit = FT(0.95))
        CUDA.synchronize()
        t_step = time() - t0
        @info @sprintf("  Trial %d: 1 Strang step = %.4fs", trial, t_step)
    end

    @info "\n--- Phase 3: Full window (24 steps) WITH m-reset ---"
    m_dev3 = CuArray(m_cpu)
    c3 = CUDA.zeros(FT, Nx, Ny, Nz) .+ FT(1e-3)
    tracers3 = (; c = c3)
    ws3 = allocate_massflux_workspace(m_dev3, am_dev, bm_dev, cm_dev)
    CUDA.synchronize()
    t0 = time()
    for sub in 1:24
        copyto!(m_dev3, m_cpu)
        strang_split_massflux!(tracers3, m_dev3, am_dev, bm_dev, cm_dev,
                               grid, true, ws3; cfl_limit = FT(0.95))
    end
    CUDA.synchronize()
    t_window = time() - t0
    @info @sprintf("  24 steps (m-reset) = %.2fs (%.4fs/step)", t_window, t_window/24)
    @info @sprintf("  Projected 120 windows = %.0fs = %.1f min", 120*t_window, 120*t_window/60)

    @info "\n--- Phase 3b: Full window (24 steps) WITHOUT m-reset (for comparison) ---"
    m_dev3b = CuArray(m_cpu)
    c3b = CUDA.zeros(FT, Nx, Ny, Nz) .+ FT(1e-3)
    tracers3b = (; c = c3b)
    ws3b = allocate_massflux_workspace(m_dev3b, am_dev, bm_dev, cm_dev)
    CUDA.synchronize()
    t0 = time()
    for sub in 1:24
        strang_split_massflux!(tracers3b, m_dev3b, am_dev, bm_dev, cm_dev,
                               grid, true, ws3b; cfl_limit = FT(0.95))
    end
    CUDA.synchronize()
    t_window_no_reset = time() - t0
    @info @sprintf("  24 steps (no reset) = %.2fs (%.4fs/step)", t_window_no_reset, t_window_no_reset/24)
    @info @sprintf("  Speedup from m-reset: %.0fx", t_window_no_reset / t_window)

    @info "\n--- Phase 4: Individual kernel timing ---"
    m_dev4 = CuArray(m_cpu)
    c4 = CUDA.zeros(FT, Nx, Ny, Nz) .+ FT(1e-3)
    ws4 = allocate_massflux_workspace(m_dev4, am_dev, bm_dev, cm_dev)
    rm = similar(c4)
    rm .= m_dev4 .* c4
    rm_t = (; c = rm)

    CUDA.synchronize()
    t0 = time()
    n_x = advect_x_massflux_subcycled!(rm_t, m_dev4, am_dev, grid, true, ws4; cfl_limit=FT(0.95))
    CUDA.synchronize()
    t_x = time() - t0

    m_dev4 = CuArray(m_cpu)
    rm .= m_dev4 .* c4
    CUDA.synchronize()
    t0 = time()
    n_y = advect_y_massflux_subcycled!(rm_t, m_dev4, bm_dev, grid, true, ws4; cfl_limit=FT(0.95))
    CUDA.synchronize()
    t_y = time() - t0

    m_dev4 = CuArray(m_cpu)
    rm .= m_dev4 .* c4
    CUDA.synchronize()
    t0 = time()
    n_z = advect_z_massflux_subcycled!(rm_t, m_dev4, cm_dev, true, ws4; cfl_limit=FT(0.95))
    CUDA.synchronize()
    t_z = time() - t0

    @info @sprintf("  x-advect: %.4fs (%d subcycles)", t_x, n_x)
    @info @sprintf("  y-advect: %.4fs (%d subcycles)", t_y, n_y)
    @info @sprintf("  z-advect: %.4fs (%d subcycles)", t_z, n_z)

    @info "\n" * "=" ^ 70
end

profile_gpu()
