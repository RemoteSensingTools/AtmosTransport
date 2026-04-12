#!/usr/bin/env julia
# Track sum(rm) through every step of the substep loop to identify
# exactly where mass leaks: transport, c=rm/m, or rm=m*c roundtrip.
#
# Runs on CPU in both Float32 and Float64.
#
# Usage: julia --project=. scripts/diagnostics/diagnose_rm_roundtrip.jl

using Printf, NCDatasets
println("Loading AtmosTransport...")
using AtmosTransport
using AtmosTransport.Architectures: CPU
using AtmosTransport.Grids: LatitudeLongitudeGrid, merge_thin_levels
using AtmosTransport.Advection: allocate_massflux_workspace, strang_split_massflux_ppm!,
    MassFluxWorkspace
println("Loaded.")

const NC_PATH = expanduser("~/data/AtmosTransport/met/era5/preprocessed_spectral_catrine/massflux_era5_spectral_202112_float32.nc")

function merge_field!(out, native, mm, d1, d2, Nz_n)
    fill!(out, zero(eltype(out)))
    for k_n in 1:Nz_n
        k_m = mm[k_n]
        for j in 1:d2, i in 1:d1
            out[i, j, k_m] += native[i, j, k_n]
        end
    end
end

function run_diagnostic(FT, n_win, n_sub_per_win)
    ds = NCDataset(NC_PATH, "r")
    Nx, Ny, Nz_native = 720, 361, 137

    # Build merged grid
    met_cfg = AtmosTransport.IO.default_met_config("era5")
    vc_native = AtmosTransport.IO.build_vertical_coordinate(met_cfg; FT, level_range=1:Nz_native)
    vc_m, mm = merge_thin_levels(vc_native; min_thickness_Pa=1000.0)
    Nz = length(vc_m.A) - 1

    grid = LatitudeLongitudeGrid(CPU(); FT, size=(Nx, Ny, Nz), vertical=vc_m)
    cs_cpu = Int32.(grid.reduced_grid.cluster_sizes)

    # Allocate arrays
    m_ref = zeros(FT, Nx, Ny, Nz)
    am = zeros(FT, Nx+1, Ny, Nz)
    bm = zeros(FT, Nx, Ny+1, Nz)
    cm = zeros(FT, Nx, Ny, Nz+1)
    ws = allocate_massflux_workspace(m_ref, am, bm, cm; cluster_sizes_cpu=cs_cpu)

    # Load real CO2 IC (from the NetCDF initial condition file)
    # Use a synthetic non-uniform IC: CO2 gradient from 390 (SH) to 410 (NH)
    c = zeros(FT, Nx, Ny, Nz)
    for j in 1:Ny
        lat = FT(-90.0 + (j - 1) * 0.5)
        c[:, j, :] .= FT(400.0 + 10.0 * lat / 90.0)  # 390-410 ppm gradient
    end

    println("\n=== rm roundtrip diagnostic ($FT) ===")
    println("Grid: $Nx × $Ny × $Nz (merged), n_sub=$n_sub_per_win")
    println()

    @printf("%-4s %-4s  %18s  %18s  %18s  %14s  %14s\n",
            "win", "sub", "Σrm_before_adv", "Σrm_after_adv", "Σrm_after_roundtrip",
            "Δ_transport(%)", "Δ_roundtrip(%)")

    initial_rm = sum(Float64, m_ref)  # will be set properly below
    cum_transport_err = 0.0
    cum_roundtrip_err = 0.0

    for w in 1:n_win
        # Load and merge met data
        m_nat = FT.(ds["m"][:, :, :, w])
        am_nat = FT.(ds["am"][:, :, :, w])
        bm_nat = FT.(ds["bm"][:, :, :, w])
        cm_nat = FT.(ds["cm"][:, :, :, w])

        merge_field!(m_ref, m_nat, mm, Nx, Ny, Nz_native)
        merge_field!(am, am_nat, mm, Nx + 1, Ny, Nz_native)
        merge_field!(bm, bm_nat, mm, Nx, Ny + 1, Nz_native)
        # cm needs recomputation from merged am/bm (continuity)
        # Use the _cm_column_kernel approach
        vc = grid.vertical
        dB = [FT(vc.B[k+1] - vc.B[k]) for k in 1:Nz]
        dB_total = FT(vc.B[Nz+1] - vc.B[1])
        bt = abs(dB_total) > eps(FT) ? dB ./ dB_total : zeros(FT, Nz)
        fill!(cm, zero(FT))
        for j in 1:Ny, i in 1:Nx
            pit = zero(FT)
            for k in 1:Nz
                pit += am[i, j, k] - am[i+1, j, k] + bm[i, j, k] - bm[i, j+1, k]
            end
            acc = zero(FT)
            cm[i, j, 1] = zero(FT)
            for k in 1:Nz
                conv_k = am[i, j, k] - am[i+1, j, k] + bm[i, j, k] - bm[i, j+1, k]
                acc += conv_k - bt[k] * pit
                cm[i, j, k+1] = acc
            end
        end

        if w == 1
            initial_rm = sum(Float64, m_ref .* c)
        end

        for s in 1:n_sub_per_win
            # Step 1: rm = m_ref * c (this is the roundtrip recomputation)
            rm_before = m_ref .* c
            sum_rm_before = sum(Float64, rm_before)

            # Step 2: run Strang split (modifies rm and m in-place)
            m_work = copy(m_ref)
            tracers = (co2 = c,)
            strang_split_massflux_ppm!(tracers, m_work, am, bm, cm, grid, Val(7), ws;
                                        cfl_limit=FT(0.95))
            # After strang_split: c = ws.rm / m_work (already done inside)
            # ws.rm contains the transported rm, m_work is the transported m
            sum_rm_after = sum(Float64, ws.rm)

            # Step 3: the roundtrip — what rm will be on next substep
            # Next substep will do: rm_next = m_ref * c = m_ref * (ws.rm / m_work)
            rm_roundtrip = m_ref .* c  # c was already set to ws.rm / m_work inside strang_split
            sum_rm_roundtrip = sum(Float64, rm_roundtrip)

            transport_err = (sum_rm_after - sum_rm_before) / sum_rm_before * 100
            roundtrip_err = (sum_rm_roundtrip - sum_rm_after) / sum_rm_after * 100
            cum_transport_err += transport_err
            cum_roundtrip_err += roundtrip_err

            total_from_init = (sum_rm_roundtrip - initial_rm) / initial_rm * 100

            @printf("%3d  %3d  %18.10e  %18.10e  %18.10e  %+14.10f  %+14.10f\n",
                    w, s, sum_rm_before, sum_rm_after, sum_rm_roundtrip,
                    transport_err, roundtrip_err)
        end
    end

    total_pct = (sum(Float64, m_ref .* c) - initial_rm) / initial_rm * 100
    println()
    @printf("Cumulative transport error:  %+.10f%%\n", cum_transport_err)
    @printf("Cumulative roundtrip error:  %+.10f%%\n", cum_roundtrip_err)
    @printf("Total drift from initial:    %+.10f%%\n", total_pct)
    println("Transport error = flux non-telescoping (should be ~0)")
    println("Roundtrip error = cov(c, m_ref/m_transport - 1) (the mass leak)")

    close(ds)
end

# Run with Float32 (matching GPU L40S)
run_diagnostic(Float32, 3, 24)

# Run with Float64 (CPU reference)
run_diagnostic(Float64, 3, 24)
