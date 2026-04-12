#!/usr/bin/env julia
# =============================================================================
# Spatial structure of hourly (or arbitrary-cadence) Δps from ERA5 spectral LNSP.
#
# For each consecutive pair of analyses, Δps_ij = ps(t+1) - ps(t). Compare:
#   - Area-weighted mean Δps̄ (spatially uniform increment).
#   - Residual r_ij = Δps_ij - Δps̄ (mass redistribution with zero global ΔM).
#
# Metrics (area-weighted, per step, then summarized):
#   ρ_res = RMS(Δps − area-mean Δps) / RMS(Δps)  — near 1 ⇒ area-mean increment is
#           tiny vs typical |Δps|; spatial redistribution dominates RMS (maps matter).
#           Near 0 ⇒ increments are almost a spatially uniform Δps.
#   ρ_lon = RMS(Δps − zonal_mean(Δps)) / RMS(Δps)  — small ⇒ little longitudinal
#           structure (zonal symmetry of the increment).
#
# Outputs: text statistics + figures (time-mean |r̄|, Hovmöller of zonal-mean Δps,
# optional map of zonal-mean time RMS).
#
# Usage:
#   julia --project=. scripts/diagnostics/spectral_lnsp_spatial_drift.jl \
#       /path/to/spectral_lnsp_dir [--out-prefix dir/prefix]
#
# RAM: stores full spectral stack (~ (T+1)² × N × 16 B). Use a short month or
# subsample the directory for quick tests.
# =============================================================================

using Printf
using Statistics
using CairoMakie

include(joinpath(@__DIR__, "spectral_lnsp_grib_utils.jl"))

function area_weighted_rms(a::Matrix{Float64}, w::Matrix{Float64})
    s = sum(w)
    iszero(s) && return 0.0
    return sqrt(sum(@. w * a^2) / s)
end

function zonal_mean(mat::Matrix{Float64}, Nlon::Int, Nlat::Int)
    z = zeros(Nlat)
    for j in 1:Nlat
        s = 0.0
        @inbounds for i in 1:Nlon
            s += mat[i, j]
        end
        z[j] = s / Nlon
    end
    return z
end

function subtract_zonal!(out::Matrix{Float64}, mat::Matrix{Float64}, z::Vector{Float64},
                         Nlon::Int, Nlat::Int)
    for j in 1:Nlat
        zj = z[j]
        @inbounds for i in 1:Nlon
            out[i, j] = mat[i, j] - zj
        end
    end
    return nothing
end

function main()
    if length(ARGS) < 1
        println("""
        Usage:
          julia --project=. $(basename(PROGRAM_FILE)) <spectral_dir> [--out-prefix path]

        Writes path_spatial_mean_abs_residual.png, path_hovmoller_zonal_dps.png, etc.
        """)
        return
    end
    spectral_dir = expanduser(ARGS[1])
    prefix = joinpath(spectral_dir, "lnsp_spatial_drift")
    for i in 1:length(ARGS)-1
        ARGS[i] == "--out-prefix" && (prefix = expanduser(ARGS[i+1]))
    end

    @info "Loading spectral coefficients (RAM-heavy)…"
    times, coeff, grid, T = collect_lnsp_coeff_stack(spectral_dir)
    N = length(times)
    N < 2 && error("Need at least two time steps")
    Nlon, Nlat = grid.Nlon, grid.Nlat
    area = grid.area

    P_buf = zeros(T + 1, T + 1)
    fft_buf = zeros(ComplexF64, Nlon)
    lnsp_grid = zeros(Nlon, Nlat)
    ps_a = zeros(Nlon, Nlat)
    ps_b = zeros(Nlon, Nlat)
    dps = zeros(Nlon, Nlat)
    r = zeros(Nlon, Nlat)
    dps_zon_res = zeros(Nlon, Nlat)

    ρ_res = Float64[]
    ρ_lon = Float64[]
    mean_abs_r = zeros(Nlon, Nlat)
    zonal_dps_stack = zeros(Nlat, N - 1)
    zonal_rms_lat = zeros(Nlat)

    wsum = sum(area)
    for k in 1:N-1
        spec_k = @view coeff[:, :, k]
        spec_k1 = @view coeff[:, :, k+1]
        lnsp_spec_to_ps!(ps_a, spec_k, T, grid, P_buf, fft_buf, lnsp_grid)
        lnsp_spec_to_ps!(ps_b, spec_k1, T, grid, P_buf, fft_buf, lnsp_grid)
        @. dps = ps_b - ps_a
        dps_mean = sum(dps .* area) / wsum
        @. r = dps - dps_mean
        Rr = area_weighted_rms(r, area)
        Rd = area_weighted_rms(dps, area)
        push!(ρ_res, iszero(Rd) ? 0.0 : Rr / Rd)

        z = zonal_mean(dps, Nlon, Nlat)
        subtract_zonal!(dps_zon_res, dps, z, Nlon, Nlat)
        Rlon = area_weighted_rms(dps_zon_res, area)
        push!(ρ_lon, iszero(Rd) ? 0.0 : Rlon / Rd)

        mean_abs_r .+= abs.(r)
        zonal_dps_stack[:, k] .= z
        for j in 1:Nlat
            zonal_rms_lat[j] += z[j]^2
        end
    end
    mean_abs_r ./= (N - 1)
    zonal_rms_lat .= sqrt.(zonal_rms_lat ./ (N - 1))

    println("\n=== Spatial structure of Δps (consecutive GRIB times) ===")
    @printf("Steps: %d   Grid: %d × %d   spectral T=%d\n", N - 1, Nlon, Nlat, T)
    @printf("Median  RMS(residual) / RMS(Δps):  %.4f\n", median(ρ_res))
    @printf("Mean    RMS(residual) / RMS(Δps):  %.4f\n", mean(ρ_res))
    @printf("Median  RMS(zonal-anom Δps) / RMS(Δps): %.4f\n", median(ρ_lon))
    @printf("Mean    RMS(zonal-anom Δps) / RMS(Δps): %.4f\n", mean(ρ_lon))
    println("""
Interpretation (qualitative):
  ρ_res ≈ 1  → subtracting area-mean Δps barely lowers RMS: spatial patterns dominate;
               global-mass jumps are a small uniform component vs weather-scale Δps.
  ρ_res ≪ 1  → almost uniform Δps field (whole atmosphere rises/falls together).
  ρ_lon ≪ 1  → nearly zonal Δps (little variation along longitude at fixed lat).
  ρ_lon ≈ 1  → strong zonal asymmetry (synoptic / geographic structure in Δps).
""")

    # --- Figures ---
    fig1 = Figure(size=(900, 400))
    axm = Axis(fig1[1, 1];
               title="Time-mean |Δps − area-mean(Δps)|  (Pa)",
               xlabel="Longitude index", ylabel="Latitude index")
    hm = heatmap!(axm, 1:Nlon, 1:Nlat, mean_abs_r'; colormap=:thermal)
    Colorbar(fig1[1, 2], hm)
    save(prefix * "_mean_abs_residual.png", fig1)
    println("Wrote ", prefix * "_mean_abs_residual.png")

    tmid = times[2:end]
    th = Float64[((t - times[1]).value) / 3_600_000 for t in tmid]
    fig2 = Figure(size=(1000, 500))
    axh = Axis(fig2[1, 1];
               title="Zonal-mean Δps (Pa) — latitude vs hours since start",
               xlabel="Hours since $(times[1]) (end of each Δt)", ylabel="Latitude (°)")
    hm2 = heatmap!(axh, th, grid.lats, zonal_dps_stack;
                   colormap=:balance)
    Colorbar(fig2[1, 2], hm2)
    save(prefix * "_hovmoller_zonal_dps.png", fig2)
    println("Wrote ", prefix * "_hovmoller_zonal_dps.png")

    fig3 = Figure(size=(700, 400))
    axz = Axis(fig3[1, 1];
               title="RMS over time of zonal-mean Δps (Pa)",
               xlabel="Latitude (°)", ylabel="RMS_zonal(Δps)")
    lines!(axz, grid.lats, zonal_rms_lat)
    save(prefix * "_zonal_rms_vs_lat.png", fig3)
    println("Wrote ", prefix * "_zonal_rms_vs_lat.png")

    fig4 = Figure(size=(700, 400))
    axr = Axis(fig4[1, 1];
               title="ρ = RMS(spatial residual) / RMS(Δps) per step",
               xlabel="Step index (pairs)", ylabel="ρ_res")
    scatter!(axr, 1:length(ρ_res), ρ_res; markersize=3)
    save(prefix * "_rho_res_timeseries.png", fig4)
    println("Wrote ", prefix * "_rho_res_timeseries.png")
end

main()
