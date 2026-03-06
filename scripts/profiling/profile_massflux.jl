#!/usr/bin/env julia
# Profile mass-flux advection: measure CFL, subcycle counts, and kernel timing

using AtmosTransport
using AtmosTransport.Architectures
using AtmosTransport.Grids
using AtmosTransport.Advection
using AtmosTransport.Parameters
using AtmosTransport.IO: default_met_config, build_vertical_coordinate,
                              load_vertical_coefficients
using NCDatasets
using Printf

const USE_GPU     = parse(Bool, get(ENV, "USE_GPU", "false"))
const USE_FLOAT32 = parse(Bool, get(ENV, "USE_FLOAT32", "true"))
if USE_GPU
    using CUDA
    CUDA.allowscalar(false)
end

const FT = USE_FLOAT32 ? Float32 : Float64

const MASSFLUX_FILE = expanduser(get(ENV, "MASSFLUX_FILE",
    "~/data/metDrivers/era5/massflux_era5_202406_$(USE_FLOAT32 ? "float32" : "float64").nc"))

function profile()
    @info "=" ^ 70
    @info "Mass-Flux Profiling — CFL and Subcycle Analysis"
    @info "=" ^ 70
    @info "  FT=$FT, arch=$(USE_GPU ? "GPU" : "CPU")"

    ds_mf = NCDataset(MASSFLUX_FILE)
    lons = FT.(ds_mf["lon"][:])
    lats = FT.(ds_mf["lat"][:])
    Nx   = length(lons)
    Ny   = length(lats)
    Nz   = ds_mf.dim["lev"]
    level_top = ds_mf.attrib["level_top"]
    level_bot = ds_mf.attrib["level_bot"]

    config = default_met_config("era5")
    vc = build_vertical_coordinate(config; FT, level_range=level_top:level_bot)
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

    m_cpu  = FT.(ds_mf["m"][:, :, :, 1])
    am_cpu = FT.(ds_mf["am"][:, :, :, 1])
    bm_cpu = FT.(ds_mf["bm"][:, :, :, 1])
    cm_cpu = FT.(ds_mf["cm"][:, :, :, 1])
    close(ds_mf)

    @info "\n--- CFL Analysis (window 1, CPU) ---"

    cfl_x = similar(am_cpu)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:(Nx+1)
        il = i == 1 ? Nx : i - 1
        ir = i > Nx ? 1 : i
        md = am_cpu[i,j,k] >= 0 ? m_cpu[max(il,1),j,k] : m_cpu[min(ir,Nx),j,k]
        cfl_x[i,j,k] = md > 0 ? abs(am_cpu[i,j,k]) / md : zero(FT)
    end
    cfl_y = similar(bm_cpu)
    @inbounds for k in 1:Nz, j in 1:(Ny+1), i in 1:Nx
        if j >= 2 && j <= Ny
            md = bm_cpu[i,j,k] >= 0 ? m_cpu[i,j-1,k] : m_cpu[i,j,k]
            cfl_y[i,j,k] = md > 0 ? abs(bm_cpu[i,j,k]) / md : zero(FT)
        else
            cfl_y[i,j,k] = zero(FT)
        end
    end
    cfl_z = similar(cm_cpu)
    @inbounds for k in 1:(Nz+1), j in 1:Ny, i in 1:Nx
        if k >= 2 && k <= Nz
            md = cm_cpu[i,j,k] > 0 ? m_cpu[i,j,k-1] : m_cpu[i,j,k]
            cfl_z[i,j,k] = md > 0 ? abs(cm_cpu[i,j,k]) / md : zero(FT)
        else
            cfl_z[i,j,k] = zero(FT)
        end
    end

    max_cfl_x_val = maximum(cfl_x)
    max_cfl_y_val = maximum(cfl_y)
    max_cfl_z_val = maximum(cfl_z)
    nsub_x = max(1, ceil(Int, max_cfl_x_val / 0.95))
    nsub_y = max(1, ceil(Int, max_cfl_y_val / 0.95))
    nsub_z = max(1, ceil(Int, max_cfl_z_val / 0.95))

    @info @sprintf("  CFL_x max = %.2f → %d subcycles", max_cfl_x_val, nsub_x)
    @info @sprintf("  CFL_y max = %.2f → %d subcycles", max_cfl_y_val, nsub_y)
    @info @sprintf("  CFL_z max = %.2f → %d subcycles", max_cfl_z_val, nsub_z)
    @info @sprintf("  Strang (X-Y-Z-Z-Y-X) total subcycles per step: %d", 2*nsub_x + 2*nsub_y + 2*nsub_z)
    @info @sprintf("  With 24 steps/window: %d kernel launches/window", 24 * (2*nsub_x + 2*nsub_y + 2*nsub_z))

    @info "\n--- Per-latitude CFL_x breakdown ---"
    @info @sprintf("  %5s  %8s  %8s  %8s  %6s", "j", "lat", "cos(φ)", "max_CFL", "nsub")
    for j in 1:Ny
        max_j = maximum(abs.(cfl_x[:, j, :]))
        if max_j > 0.5
            lat = grid.φᶜ_cpu[j]
            nsub_j = max(1, ceil(Int, max_j / 0.95))
            @info @sprintf("  %5d  %8.2f  %8.5f  %8.2f  %6d", j, lat, cosd(Float64(lat)), max_j, nsub_j)
        end
    end

    rg = grid.reduced_grid
    if rg !== nothing
        @info "\n--- Reduced Grid Spec ---"
        @info @sprintf("  %5s  %8s  %8s  %8s  %10s", "j", "lat", "cluster", "Nx_red", "eff_CFL")
        for j in 1:Ny
            if rg.cluster_sizes[j] > 1
                lat = grid.φᶜ_cpu[j]
                max_j = maximum(abs.(cfl_x[:, j, :]))
                eff_cfl = max_j / rg.cluster_sizes[j]
                @info @sprintf("  %5d  %8.2f  %8d  %8d  %10.2f", j, lat, rg.cluster_sizes[j], rg.reduced_counts[j], eff_cfl)
            end
        end
    end

    if USE_GPU
        @info "\n--- GPU kernel timing (single step) ---"
        m_dev  = AT(m_cpu)
        am_dev = AT(am_cpu)
        bm_dev = AT(bm_cpu)
        cm_dev = AT(cm_cpu)
        c      = AT(zeros(FT, Nx, Ny, Nz))
        tracers = (; c)
        ws = allocate_massflux_workspace(m_dev, am_dev, bm_dev, cm_dev)

        @info "  Warming up GPU kernels..."
        strang_split_massflux!(tracers, m_dev, am_dev, bm_dev, cm_dev,
                               grid, true, ws; cfl_limit = FT(0.95))
        CUDA.synchronize()

        @info "  Timing 1 step..."
        t0 = time()
        strang_split_massflux!(tracers, m_dev, am_dev, bm_dev, cm_dev,
                               grid, true, ws; cfl_limit = FT(0.95))
        CUDA.synchronize()
        t1 = time() - t0
        @info @sprintf("  1 Strang step = %.3fs", t1)
        @info @sprintf("  Projected 24 steps/window = %.1fs", 24 * t1)
        @info @sprintf("  Projected 120 windows = %.0fs = %.1f min", 120 * 24 * t1, 120 * 24 * t1 / 60)
    end

    @info "\n" * "=" ^ 70
end

profile()
