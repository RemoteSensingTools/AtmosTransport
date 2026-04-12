#!/usr/bin/env julia
# =============================================================================
# Time series of global dry-air mass implied by ERA5 spectral LNSP (GRIB).
#
# Matches preprocess_spectral_v4_binary.jl: TargetGrid(720,361), T = Nlon/2 - 1,
# spectral_to_grid! with lon_shift = dlon/2, ps = exp(LNSP), then
#
#   M_tot = sum_ij ps_ij * area_ij / g
#
# which is the standard column-mass integral ∫∫ (ps/g) dA for a hydrostatic
# atmosphere with surface pressure ps (ERA5 hybrid sums to ps per column).
#
# Usage:
#   julia --project=. scripts/diagnostics/spectral_lnsp_total_mass_timeseries.jl \
#       /path/to/met/era5/spectral [--out figure.png]
#
# Notes:
# - One (time, M) sample per GRIB message, sorted by validity time. Hourly spectral
#   LNSP → hourly M(t); ΔM and Δt use consecutive samples (Δt typically 1 h).
# - Coarser pulls (e.g. default download_era5_spectral.py at 00/06/12/18 UTC only)
#   are the same code path — fewer messages per day, larger Δt between points.
# =============================================================================

using Printf
using Statistics
using CairoMakie

include(joinpath(@__DIR__, "spectral_lnsp_grib_utils.jl"))

function main()
    if length(ARGS) < 1
        println("""
        Usage:
          julia --project=. $(basename(PROGRAM_FILE)) <spectral_dir> [--out path.png]

        Computes M = sum(ps*area)/g from spectral LNSP (v4-consistent grid/transform).
        """)
        return
    end
    spectral_dir = expanduser(ARGS[1])
    outpath = nothing
    for i in 2:length(ARGS)-1
        ARGS[i] == "--out" && (outpath = expanduser(ARGS[i+1]))
    end

    series, T = collect_lnsp_series(spectral_dir)
    times = [s[1] for s in series]
    M = [s[2] for s in series]
    @printf("Samples: %d  (spectral T_used=%d)\n", length(times), T)
    @printf("Time range: %s  →  %s\n", times[1], times[end])
    hrs = [hour(t) + minute(t) / 60 for t in times]
    uh = unique(hrs)
    @printf("Unique UTC hours in data: %s\n",
            join(sort!(round.(uh, digits=2)), ", "))

    ΔM = diff(M)
    Δt_h = [(times[i+1] - times[i]).value / 3600000.0 for i in 1:length(times)-1]
    dm_dt = ΔM ./ Δt_h

    M0 = M[1]
    @printf("\nM_total first / last: %.6e / %.6e kg\n", M[1], M[end])
    @printf("ΔM total (last - first): %.6e kg (%.4f %% of M0)\n",
            M[end] - M0, 100 * (M[end] - M0) / M0)

    buckets = Dict{Int,Vector{Float64}}()
    for i in 1:length(ΔM)
        h2 = hour(times[i+1])
        push!(get!(()->Float64[], buckets, h2), ΔM[i])
    end
    println("\nPer analysis hour (later time of each pair): mean ΔM [kg], std, N")
    for h in sort(collect(keys(buckets)))
        v = buckets[h]
        σ = length(v) > 1 ? std(v) : 0.0
        @printf("  %02d UTC: mean=% .4e  std=% .4e  N=%d\n", h, mean(v), σ, length(v))
    end

    into_main = Int[]
    into_filt = Int[]
    for i in 1:length(ΔM)
        h2 = hour(times[i+1])
        if h2 in (0, 12)
            push!(into_main, i)
        elseif h2 in (6, 18)
            push!(into_filt, i)
        end
    end
    if !isempty(into_main) && !isempty(into_filt)
        a = mean(abs.(ΔM[into_main]))
        b = mean(abs.(ΔM[into_filt]))
        @printf("\nMean |ΔM| when next sample is 00/12 UTC: %.4e kg\n", a)
        @printf("Mean |ΔM| when next sample is 06/18 UTC: %.4e kg\n", b)
        @printf("Ratio (00/12)/(06/18): %.3f\n", a / b)
    end

    if outpath === nothing
        outpath = joinpath(@__DIR__, "spectral_lnsp_mass_timeseries.png")
    end

    fig = Figure(size=(900, 700))
    ax1 = Axis(fig[1, 1];
               title="Global mass from LNSP (v4 grid/transform)",
               xlabel="Time (UTC)",
               ylabel="M - M(t₀)  (10¹⁵ kg)")
    lines!(ax1, times, (M .- M0) ./ 1e15)
    ax2 = Axis(fig[2, 1];
               title="Step change ΔM / Δt (proxy for implied mass tendency)",
               xlabel="Time (UTC) — point at end of interval",
               ylabel="dM/dt  (10¹² kg / h)")
    if !isempty(dm_dt)
        lines!(ax2, times[2:end], dm_dt ./ 1e12)
    end
    save(outpath, fig)
    println("\nWrote ", outpath)
end

main()
