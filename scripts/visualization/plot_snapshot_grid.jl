#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# plot_snapshot_grid.jl — plot all snapshots in a single multi-panel PNG
#
# Works on both LL (lon, lat, time) snapshots and CS (Xdim, Ydim, nf, lev,
# time) snapshots. CS input is air-mass-weighted column-mean dry-VMR and
# is conservatively regridded to LL for plotting (ConservativeRegridding,
# same path as compare_cs_vs_ll.jl). LL input is plotted directly.
#
# Usage:
#   julia --project=. scripts/visualization/plot_snapshot_grid.jl \
#       --input  ~/data/AtmosTransport/output/catrine_c48_3d/advonly_cpu_float32.nc \
#       --tracer co2_natural \
#       --out    artifacts/plan40/c48_advonly_cpu_f32_natural.png
#       [--cols 4]        # grid layout (default 4 columns, rows inferred)
#       [--ppm]           # plot in ppm (×1e6) instead of mol/mol
# ---------------------------------------------------------------------------

using Printf
using Statistics: mean
using NCDatasets
using CairoMakie

include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.Grids: LatLonMesh, CubedSphereMesh, GnomonicPanelConvention
using .AtmosTransport.Regridding: build_regridder, apply_regridder!

const R_EARTH_M = 6.371229e6

const USAGE = """
Usage: julia --project=. scripts/visualization/plot_snapshot_grid.jl \\
           --input <snap.nc> --tracer <name> --out <out.png>
           [--cols N] [--ppm]
"""

function _parse_args(argv)
    input = nothing; tracer = nothing; out = nothing
    cols = 4; ppm = false
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--input" && i + 1 <= length(argv)
            input = expanduser(argv[i + 1]); i += 2
        elseif a == "--tracer" && i + 1 <= length(argv)
            tracer = argv[i + 1]; i += 2
        elseif a == "--out" && i + 1 <= length(argv)
            out = expanduser(argv[i + 1]); i += 2
        elseif a == "--cols" && i + 1 <= length(argv)
            cols = parse(Int, argv[i + 1]); i += 2
        elseif a == "--ppm"
            ppm = true; i += 1
        elseif a in ("-h", "--help")
            println(USAGE); exit(0)
        else
            error("Unknown arg `$(a)`.\n$(USAGE)")
        end
    end
    input  === nothing && error("--input required.\n$(USAGE)")
    tracer === nothing && error("--tracer required.\n$(USAGE)")
    out    === nothing && error("--out required.\n$(USAGE)")
    isfile(input) || error("Input NetCDF not found: $(input)")
    return (; input, tracer, out, cols, ppm)
end

# Air-mass-weighted column-mean dry VMR on CS. `vmr` + `m` have shape
# (Nc, Nc, nf, Nz, ntime). Output: (Nc, Nc, nf, ntime).
function _cs_column_mean_dry_vmr(vmr::Array{T, 5}, m::Array{T, 5}) where T
    Nc1, Nc2, nf, Nz, ntime = size(vmr)
    out = zeros(Float64, Nc1, Nc2, nf, ntime)
    @inbounds for t in 1:ntime, p in 1:nf, j in 1:Nc2, i in 1:Nc1
        num = 0.0; den = 0.0
        for k in 1:Nz
            num += Float64(vmr[i, j, p, k, t]) * Float64(m[i, j, p, k, t])
            den += Float64(m[i, j, p, k, t])
        end
        out[i, j, p, t] = den > 0 ? num / den : 0.0
    end
    return out
end

# Detect snapshot shape by looking at dimensions.
_is_cs_snapshot(ds) = haskey(ds.dim, "Xdim") && haskey(ds.dim, "nf")
_is_ll_snapshot(ds) = haskey(ds.dim, "lon") && haskey(ds.dim, "lat")

# Load column-mean VMR on a lat-lon grid, plus the time axis in hours.
# Returns (lons, lats, times_h, data[lon, lat, ntime]).
function _load_as_latlon(path::AbstractString, tracer::AbstractString)
    ds = NCDataset(path, "r")
    try
        if _is_ll_snapshot(ds)
            lons = Float64.(collect(ds["lon"][:]))
            lats = Float64.(collect(ds["lat"][:]))
            times = Float64.(collect(ds["time"][:]))
            key = "$(tracer)_column_mean"
            haskey(ds, key) || error("LL snapshot has no `$(key)`; keys = $(collect(keys(ds)))")
            data = Array{Float64}(ds[key][:, :, :])    # (Nx, Ny, ntime)
            return lons, lats, times, data
        elseif _is_cs_snapshot(ds)
            Nc = Int(ds.attrib["Nc"])
            times = Float64.(collect(ds["time"][:]))
            haskey(ds, tracer) || error("CS snapshot has no `$(tracer)`; keys = $(collect(keys(ds)))")
            vmr_full = Array{Float64}(ds[tracer][:, :, :, :, :])
            air = Array{Float64}(ds["air_mass"][:, :, :, :, :])
            cs_cm = _cs_column_mean_dry_vmr(vmr_full, air)     # (Nc, Nc, nf, ntime)
            # Assert gnomonic source. The CS writer now records
            # `panel_convention` (plan 40 Commit 7); a GEOS-native file would
            # be silently mis-oriented by a gnomonic source mesh, so refuse
            # explicitly (same contract as `compare_cs_vs_ll.jl`).
            conv = String(get(ds.attrib, "panel_convention", "gnomonic"))
            conv == "gnomonic" || error(
                "$(path): panel_convention=$(conv); plot_snapshot_grid.jl " *
                "builds a gnomonic source mesh for the CS→LL regrid. " *
                "Regenerate the run on gnomonic or extend this script.")

            # Target LL grid for visualization: ~1° × 1° cell-centered so
            # mesh.φᶜ lines up with the Makie heatmap axis. `Ny=180` with
            # `latitude=(-90,90)` produces φᶜ at ±89.5..±0.5 (Δφ=1°); passing
            # `-89.5..89.5` directly would over-constrain and drift away
            # from `LatLonMesh`'s actual cell centers, so we read φᶜ/λᶜ
            # back from the constructed mesh.
            Nx, Ny = 360, 180
            ll_mesh = LatLonMesh(; FT = Float64, Nx = Nx, Ny = Ny,
                                   longitude = (-180.0, 180.0),
                                   latitude  = (-90.0, 90.0),
                                   radius = R_EARTH_M)
            ll_lons = Float64.(ll_mesh.λᶜ)
            ll_lats = Float64.(ll_mesh.φᶜ)
            cs_mesh = CubedSphereMesh(; FT = Float64, Nc = Nc,
                                       radius = R_EARTH_M,
                                       convention = GnomonicPanelConvention())
            # Reverse direction: CS → LL. CR.jl's `build_regridder` accepts
            # either order; we pass CS as src, LL as dst.
            reg = build_regridder(cs_mesh, ll_mesh; normalize = false)
            ntime = length(times)
            out = Array{Float64}(undef, Nx, Ny, ntime)
            buf_dst = zeros(Float64, Nx * Ny)
            for ti in 1:ntime
                src_flat = vec(view(cs_cm, :, :, :, ti))  # (Nc*Nc*6,)
                fill!(buf_dst, 0.0)
                apply_regridder!(buf_dst, reg, src_flat)
                out[:, :, ti] = reshape(buf_dst, Nx, Ny)
            end
            return ll_lons, ll_lats, times, out
        else
            error("Snapshot shape at $(path) is neither LL (lon,lat) nor CS (Xdim,nf)")
        end
    finally
        close(ds)
    end
end

function main()
    opts = _parse_args(ARGS)
    lons, lats, times, data = _load_as_latlon(opts.input, opts.tracer)
    ntime = length(times)

    scale = opts.ppm ? 1.0e6 : 1.0
    units = opts.ppm ? "ppm" : "mol mol-1"
    data_plot = data .* scale

    # Shared color scale across all snapshots for a fair comparison.
    finite = filter(isfinite, data_plot)
    vmin = minimum(finite); vmax = maximum(finite)
    @info @sprintf("Snapshots: %d  range: [%.4g, %.4g] %s", ntime, vmin, vmax, units)

    cols = max(1, opts.cols)
    rows = ceil(Int, ntime / cols)

    fig = Figure(size = (250 * cols, 160 * rows + 120),
                 title = "$(opts.tracer) column-mean — $(basename(opts.input))")
    axes = Axis[]
    hm = nothing
    for ti in 1:ntime
        r = div(ti - 1, cols) + 1
        c = mod(ti - 1, cols) + 1
        ax = Axis(fig[r, c]; aspect = DataAspect(),
                  title = @sprintf("t = %.0f h", times[ti]),
                  xticksvisible = false, yticksvisible = false,
                  xticklabelsvisible = false, yticklabelsvisible = false)
        h = heatmap!(ax, lons, lats, data_plot[:, :, ti];
                    colorrange = (vmin, vmax), colormap = :viridis)
        hm === nothing && (hm = h)
        push!(axes, ax)
    end
    if hm !== nothing
        Colorbar(fig[1:rows, cols + 1], hm; label = "$(opts.tracer) [$units]",
                 height = Relative(0.9))
    end

    mkpath(dirname(opts.out))
    save(opts.out, fig)
    @info "Saved $(opts.out)"
    return opts.out
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
