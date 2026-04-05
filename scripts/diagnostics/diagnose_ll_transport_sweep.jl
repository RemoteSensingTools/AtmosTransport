#!/usr/bin/env julia
# Instrumented LL transport diagnostic: per-directional-sweep min(m), max CFL, first NaN
# Transport-only: no emissions, no diffusion, no chemistry
# Uses vertically varying IC (from CATRINE) but no surface fluxes

using CUDA
using AtmosTransport
using AtmosTransport.Architectures: GPU
using AtmosTransport.IO: MassFluxBinaryReader, load_window!
using AtmosTransport.Grids: LatitudeLongitudeGrid, HybridSigmaPressure
using AtmosTransport.Advection: allocate_massflux_workspace,
    advect_x_massflux_subcycled!, advect_y_massflux_subcycled!,
    advect_z_massflux_subcycled!, advect_z_massflux!
using Statistics, Printf, NCDatasets

function diagnose()
    # --- Load binary ---
    bin_file = "/temp1/atmos_transport/era5_daily_v3/era5_v3_20211201_merged1000Pa_float32.bin"
    reader = MassFluxBinaryReader(bin_file, Float32)
    Nx, Ny, Nz = reader.Nx, reader.Ny, reader.Nz
    FT = Float32

    # CPU buffers
    m_cpu  = zeros(FT, Nx, Ny, Nz)
    am_cpu = zeros(FT, Nx+1, Ny, Nz)
    bm_cpu = zeros(FT, Nx, Ny+1, Nz)
    cm_cpu = zeros(FT, Nx, Ny, Nz+1)
    ps_cpu = zeros(FT, Nx, Ny)

    # Build grid
    vc = HybridSigmaPressure(Float32.(reader.A_ifc), Float32.(reader.B_ifc))
    grid = LatitudeLongitudeGrid(GPU(); FT=Float32, size=(Nx,Ny,Nz), vertical=vc)

    # Workspace
    cs_cpu = grid.reduced_grid !== nothing ? Int32.(grid.reduced_grid.cluster_sizes) : ones(Int32, Ny)
    am_ws = CUDA.zeros(FT, Nx+1, Ny, Nz)
    bm_ws = CUDA.zeros(FT, Nx, Ny+1, Nz)
    cm_ws = CUDA.zeros(FT, Nx, Ny, Nz+1)
    ws = allocate_massflux_workspace(CUDA.zeros(FT,Nx,Ny,Nz), am_ws, bm_ws, cm_ws;
                                      cluster_sizes_cpu=cs_cpu)

    # --- Load IC: vertically varying CO2 from CATRINE ---
    ic_path = expanduser("~/data/AtmosTransport/catrine/InitialConditions/startCO2_202112010000.nc")
    ds_ic = NCDataset(ic_path, "r")
    co2_ic = Float64.(ds_ic["CO2"][:,:,:])  # (360, 180, 79) dry VMR
    close(ds_ic)

    # Load window 1 for m
    load_window!(m_cpu, am_cpu, bm_cpu, cm_cpu, ps_cpu, reader, 1)

    # Zero pole am (runtime guard)
    am_cpu[:, 1, :] .= 0f0
    am_cpu[:, Ny, :] .= 0f0

    # GPU arrays
    m_gpu = CuArray(m_cpu)
    am_gpu = CuArray(am_cpu)
    bm_gpu = CuArray(bm_cpu)
    cm_gpu = CuArray(cm_cpu)

    # Simple IC: vertically varying, horizontally uniform
    # Use column mean of the IC at each level (eliminates horizontal structure)
    co2_profile = Float32.([mean(co2_ic[:,:,k]) for k in 1:size(co2_ic,3)])
    # Interpolate from 79 levels to our 68 levels (simple nearest-neighbor for diagnostic)
    co2_68 = zeros(Float32, Nz)
    for k in 1:Nz
        k_src = clamp(round(Int, k * size(co2_ic,3) / Nz), 1, size(co2_ic,3))
        co2_68[k] = co2_profile[k_src]
    end
    @info "IC profile: surface=$(co2_68[end]*1e6) ppm, TOA=$(co2_68[1]*1e6) ppm"

    # Create rm = c × m (uniform horizontally, varying vertically)
    rm_cpu = zeros(FT, Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        rm_cpu[i,j,k] = co2_68[k] * m_cpu[i,j,k]
    end
    rm_gpu = CuArray(rm_cpu)
    m_dev = copy(m_gpu)

    # --- Diagnostic functions ---
    function report(label, rm, m)
        rm_h = Array(rm)
        m_h = Array(m)
        c_h = rm_h ./ m_h .* 1f6
        min_m = minimum(m_h)
        min_rm = minimum(rm_h)
        n_neg_m = sum(m_h .< 0)
        n_neg_rm = sum(rm_h .< 0)
        n_nan = sum(isnan.(rm_h)) + sum(isnan.(m_h))
        # SH Pacific std (key metric)
        lats = collect(range(-89.75f0, 89.75f0, length=Ny))
        jm = findall(-55f0 .< lats .< -45f0)
        pac_std = std(c_h[300:400, jm, Nz])  # mid-Pacific surface
        @printf("  %-25s min_m=%.2e  min_rm=%.2e  neg_m=%d  neg_rm=%d  NaN=%d  Pacific_sfc_std=%.4f ppm\n",
                label, min_m, min_rm, n_neg_m, n_neg_rm, n_nan, pac_std)
        if n_nan > 0
            idx = findfirst(isnan, rm_h)
            ci = CartesianIndices(rm_h)[idx]
            @printf("    FIRST NaN at (%d, %d, %d) lat≈%.1f°\n", ci[1], ci[2], ci[3], lats[ci[2]])
        end
        if n_neg_m > 0
            idx = argmin(m_h)
            ci = CartesianIndices(m_h)[idx]
            @printf("    MOST NEG m at (%d, %d, %d) lat≈%.1f° val=%.2e\n",
                    ci[1], ci[2], ci[3], lats[ci[2]], m_h[ci])
        end
    end

    println("="^100)
    println("INSTRUMENTED LL TRANSPORT: per-sweep diagnostics, evolving m_dev, no reset")
    println("="^100)

    tracers = (; co2 = rm_gpu)
    report("INITIAL", rm_gpu, m_dev)

    # --- Substep 1: manual Strang X-Y-Z-Z-Y-X ---
    @info "--- Substep 1 ---"

    advect_x_massflux_subcycled!(tracers, m_dev, am_gpu, grid, true, ws; cfl_limit=0.95f0)
    CUDA.synchronize()
    report("after X (1st)", rm_gpu, m_dev)

    advect_y_massflux_subcycled!(tracers, m_dev, bm_gpu, grid, true, ws; cfl_limit=0.95f0)
    CUDA.synchronize()
    report("after Y (1st)", rm_gpu, m_dev)

    advect_z_massflux_subcycled!(tracers, m_dev, cm_gpu, true, ws; cfl_limit=0.95f0)
    CUDA.synchronize()
    report("after Z (1st)", rm_gpu, m_dev)

    advect_z_massflux_subcycled!(tracers, m_dev, cm_gpu, true, ws; cfl_limit=0.95f0)
    CUDA.synchronize()
    report("after Z (2nd)", rm_gpu, m_dev)

    advect_y_massflux_subcycled!(tracers, m_dev, bm_gpu, grid, true, ws; cfl_limit=0.95f0)
    CUDA.synchronize()
    report("after Y (2nd)", rm_gpu, m_dev)

    advect_x_massflux_subcycled!(tracers, m_dev, am_gpu, grid, true, ws; cfl_limit=0.95f0)
    CUDA.synchronize()
    report("after X (2nd) = full Strang", rm_gpu, m_dev)

    # --- Substeps 2-4 ---
    for sub in 2:4
        @info "--- Substep $sub ---"
        advect_x_massflux_subcycled!(tracers, m_dev, am_gpu, grid, true, ws; cfl_limit=0.95f0)
        advect_y_massflux_subcycled!(tracers, m_dev, bm_gpu, grid, true, ws; cfl_limit=0.95f0)
        advect_z_massflux_subcycled!(tracers, m_dev, cm_gpu, true, ws; cfl_limit=0.95f0)
        advect_z_massflux_subcycled!(tracers, m_dev, cm_gpu, true, ws; cfl_limit=0.95f0)
        advect_y_massflux_subcycled!(tracers, m_dev, bm_gpu, grid, true, ws; cfl_limit=0.95f0)
        advect_x_massflux_subcycled!(tracers, m_dev, am_gpu, grid, true, ws; cfl_limit=0.95f0)
        CUDA.synchronize()
        report("after substep $sub", rm_gpu, m_dev)
    end

    println("="^100)
    println("DONE")
end

diagnose()
