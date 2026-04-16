#!/usr/bin/env julia
#
# Standalone CS transport runner — runs CS advection on C90 binary
# and exports snapshot NetCDF files for visualization.
#
# Supports TOML config files (like the LL/RG runner) or legacy --binary args.
#
# Usage (TOML):
#   julia --project=. scripts/run_cs_transport.jl config/runs/catrine_2day_cs_upwind.toml
#
# Usage (legacy):
#   julia --project=. scripts/run_cs_transport.jl \
#       --binary <cs_binary.bin> --output <snapshot.nc> \
#       [--hours 0,6,12,24] [--day2 <binary2.bin>]

using Printf
using NCDatasets
using TOML

include(joinpath(@__DIR__, "..", "src", "AtmosTransport.jl"))
using .AtmosTransport
using .AtmosTransport.Operators.Advection: fill_panel_halos!, copy_corners!,
    strang_split_cs!, CSAdvectionWorkspace,
    LinRoodWorkspace, fv_tp_2d_cs!, strang_split_linrood_ppm!,
    UpwindScheme, SlopesScheme, PPMScheme

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function parse_args()
    args = Dict{String, String}()
    i = 1
    while i <= length(ARGS)
        if startswith(ARGS[i], "--")
            key = ARGS[i][3:end]
            val = i < length(ARGS) ? ARGS[i+1] : ""
            args[key] = val
            i += 2
        else
            i += 1
        end
    end
    return args
end

"""Gnomonic projection (ξ, η) → (x, y, z) on the unit sphere for panel `p`."""
function _gnomonic_xyz(ξ::FT, η::FT, p::Int) where FT
    d = one(FT) / sqrt(one(FT) + ξ^2 + η^2)
    if     p == 1;  return ( d,  ξ*d,  η*d)
    elseif p == 2;  return (-ξ*d,  d,  η*d)
    elseif p == 3;  return (-d, -ξ*d,  η*d)
    elseif p == 4;  return ( ξ*d, -d,  η*d)
    elseif p == 5;  return (-η*d,  ξ*d,  d)
    else;           return ( η*d,  ξ*d, -d)
    end
end

"""Convert (x, y, z) on unit sphere to (lon, lat) in degrees, lon ∈ [0, 360)."""
function _xyz_to_lonlat(x, y, z)
    lon = atand(y, x)
    lat = asind(z / sqrt(x^2 + y^2 + z^2))
    lon < 0 && (lon += 360)
    return lon, lat
end

"""Panel cell centers in lat-lon degrees (for viz). Returns `(lons, lats)` each `(Nc, Nc)`."""
function panel_cell_center_lonlat(Nc::Int, p::Int, FT::Type{<:AbstractFloat})
    dα = FT(π) / (2 * Nc)
    α_centers = [FT(-π/4) + (i - 0.5) * dα for i in 1:Nc]
    lons = zeros(FT, Nc, Nc)
    lats = zeros(FT, Nc, Nc)
    for j in 1:Nc, i in 1:Nc
        x, y, z = _gnomonic_xyz(tan(α_centers[i]), tan(α_centers[j]), p)
        lons[i, j], lats[i, j] = _xyz_to_lonlat(x, y, z)
    end
    return lons, lats
end

"""Panel cell corners in lat-lon degrees. Returns `(lons, lats)` each `(Nc+1, Nc+1)`."""
function panel_cell_corner_lonlat(Nc::Int, p::Int, FT::Type{<:AbstractFloat})
    dα = FT(π) / (2 * Nc)
    α_faces = [FT(-π/4) + (i - 1) * dα for i in 1:(Nc + 1)]
    lons = zeros(FT, Nc + 1, Nc + 1)
    lats = zeros(FT, Nc + 1, Nc + 1)
    for j in 1:(Nc + 1), i in 1:(Nc + 1)
        x, y, z = _gnomonic_xyz(tan(α_faces[i]), tan(α_faces[j]), p)
        lons[i, j], lats[i, j] = _xyz_to_lonlat(x, y, z)
    end
    return lons, lats
end

"""Pad a (Nc, Nc, Nz) array to (Nc+2Hp, Nc+2Hp, Nz) with zero halo."""
function pad_panel(a, Hp)
    Nc_i, Nc_j, Nz = size(a)
    p = zeros(eltype(a), Nc_i + 2Hp, Nc_j + 2Hp, Nz)
    p[Hp+1:Hp+Nc_i, Hp+1:Hp+Nc_j, :] .= a
    return p
end

"""Pad X-direction flux: (Nc+1, Nc, Nz) → (Nc+1+2Hp, Nc+2Hp, Nz)."""
function pad_am(a, Hp)
    Nx, Ny, Nz = size(a)
    p = zeros(eltype(a), Nx + 2Hp, Ny + 2Hp, Nz)
    p[Hp+1:Hp+Nx, Hp+1:Hp+Ny, :] .= a
    return p
end

"""Pad Y-direction flux: (Nc, Nc+1, Nz) → (Nc+2Hp, Nc+1+2Hp, Nz)."""
function pad_bm(a, Hp)
    Nx, Ny, Nz = size(a)
    p = zeros(eltype(a), Nx + 2Hp, Ny + 2Hp, Nz)
    p[Hp+1:Hp+Nx, Hp+1:Hp+Ny, :] .= a
    return p
end

"""Pad Z-direction flux: (Nc, Nc, Nz+1) → (Nc+2Hp, Nc+2Hp, Nz+1)."""
function pad_cm(a, Hp)
    Nx, Ny, Nzp = size(a)
    p = zeros(eltype(a), Nx + 2Hp, Ny + 2Hp, Nzp)
    p[Hp+1:Hp+Nx, Hp+1:Hp+Ny, :] .= a
    return p
end

# ---------------------------------------------------------------------------
# Snapshot export
# ---------------------------------------------------------------------------

"""
    export_snapshot(nc_path, cs_lons, cs_lats, snap_data, snap_m,
                    snapshot_hours, Nc, Nz)

Write cubed-sphere snapshots to a GCHP/MAPL-compatible NetCDF file.

Output format matches GCHP `CATRINE_inst` files that Panoply renders natively:
- Dimensions: `(Xdim, Ydim, nf, time)` for 2D fields
- Coordinate variables: `lons(Xdim, Ydim, nf)`, `lats(Xdim, Ydim, nf)`,
  `corner_lons(XCdim, YCdim, nf)`, `corner_lats(XCdim, YCdim, nf)`
- Fake `Xdim`/`Ydim` for GrADS; `nf` with `axis="e"`
- Data variables carry `coordinates = "lons lats"` attribute
"""
function export_snapshot(nc_path::String, cs_lons, cs_lats,
                         snap_data::Vector,  # Vector of Dict{Symbol, NTuple{6, Array}}
                         snap_m::Vector,     # Vector of NTuple{6, Array} for air mass
                         snapshot_hours::Vector{Float64}, Nc::Int, Nz::Int)
    ntime = length(snap_data)

    # Compute corner coordinates for each panel
    cs_corner_lons = [panel_cell_corner_lonlat(Nc, p, Float64)[1] for p in 1:6]
    cs_corner_lats = [panel_cell_corner_lonlat(Nc, p, Float64)[2] for p in 1:6]

    isfile(nc_path) && rm(nc_path)
    mkpath(dirname(nc_path))
    ds = NCDataset(nc_path, "c")

    # Dimensions — match GCHP/MAPL convention
    defDim(ds, "Xdim", Nc)
    defDim(ds, "Ydim", Nc)
    defDim(ds, "nf", 6)
    defDim(ds, "XCdim", Nc + 1)
    defDim(ds, "YCdim", Nc + 1)
    defDim(ds, "time", ntime)

    # Global attributes
    ds.attrib["Conventions"] = "CF"
    ds.attrib["grid_mapping_name"] = "gnomonic cubed-sphere"
    ds.attrib["Source"] = "AtmosTransport.jl"
    ds.attrib["Nc"] = Nc

    # Fake Xdim / Ydim coordinate variables (for GrADS / Panoply)
    v_xdim = defVar(ds, "Xdim", Float64, ("Xdim",),
                     attrib=Dict("long_name" => "Fake Longitude for GrADS Compatibility",
                                 "units" => "degrees_east"))
    v_ydim = defVar(ds, "Ydim", Float64, ("Ydim",),
                     attrib=Dict("long_name" => "Fake Latitude for GrADS Compatibility",
                                 "units" => "degrees_north"))
    # Use panel-1 center coordinates as fake values (same as GCHP)
    v_xdim[:] = cs_lons[1][:, 1]
    v_ydim[:] = cs_lats[1][1, :]

    # nf coordinate
    v_nf = defVar(ds, "nf", Int32, ("nf",),
                   attrib=Dict("long_name" => "cubed-sphere face",
                               "axis" => "e",
                               "grads_dim" => "e"))
    v_nf[:] = Int32[1, 2, 3, 4, 5, 6]

    # Real lon/lat coordinates — cell centers (Xdim, Ydim, nf)
    v_lons = defVar(ds, "lons", Float64, ("Xdim", "Ydim", "nf"),
                     attrib=Dict("long_name" => "longitude",
                                 "units" => "degrees_east"))
    v_lats = defVar(ds, "lats", Float64, ("Xdim", "Ydim", "nf"),
                     attrib=Dict("long_name" => "latitude",
                                 "units" => "degrees_north"))
    for p in 1:6
        v_lons[:, :, p] = cs_lons[p]
        v_lats[:, :, p] = cs_lats[p]
    end

    # Corner coordinates (XCdim, YCdim, nf)
    v_clons = defVar(ds, "corner_lons", Float64, ("XCdim", "YCdim", "nf"),
                      attrib=Dict("long_name" => "longitude",
                                  "units" => "degrees_east"))
    v_clats = defVar(ds, "corner_lats", Float64, ("XCdim", "YCdim", "nf"),
                      attrib=Dict("long_name" => "latitude",
                                  "units" => "degrees_north"))
    for p in 1:6
        v_clons[:, :, p] = cs_corner_lons[p]
        v_clats[:, :, p] = cs_corner_lats[p]
    end

    # Time coordinate
    v_time = defVar(ds, "time", Float64, ("time",),
                     attrib=Dict("long_name" => "time",
                                 "units" => "hours since start"))
    v_time[:] = snapshot_hours[1:ntime]

    # Get tracer names from first snapshot
    tracer_names = collect(keys(first(snap_data)))

    # Data variables: (Xdim, Ydim, nf, time)
    data_attribs = Dict("units" => "mol mol-1 dry",
                        "coordinates" => "lons lats",
                        "grid_mapping" => "cubed_sphere")

    for name in tracer_names
        col_var = defVar(ds, "$(name)_column_mean", Float64, ("Xdim", "Ydim", "nf", "time"),
                         attrib=merge(data_attribs,
                                      Dict("long_name" => "Column-mean $name VMR")))
        sfc_var = defVar(ds, "$(name)_surface", Float64, ("Xdim", "Ydim", "nf", "time"),
                         attrib=merge(data_attribs,
                                      Dict("long_name" => "Surface $name VMR")))

        for t in 1:ntime
            rm_panels = snap_data[t][name]
            m_panels  = snap_m[t]
            for p in 1:6
                col_mean = zeros(Float64, Nc, Nc)
                surface  = zeros(Float64, Nc, Nc)
                for j in 1:Nc, i in 1:Nc
                    m_col = 0.0; rm_col = 0.0
                    for k in 1:Nz
                        m_col += m_panels[p][i, j, k]
                        rm_col += rm_panels[p][i, j, k]
                    end
                    col_mean[i, j] = m_col > 0 ? rm_col / m_col : 0.0
                    m_sfc = m_panels[p][i, j, Nz]
                    rm_sfc = rm_panels[p][i, j, Nz]
                    surface[i, j] = m_sfc > 0 ? rm_sfc / m_sfc : 0.0
                end
                col_var[:, :, p, t] = col_mean
                sfc_var[:, :, p, t] = surface
            end
        end
    end

    close(ds)
    println("Saved: $nc_path (C$Nc × 6 panels × $ntime snapshots, $(length(tracer_names)) tracers)")
end

# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

function run_cs_toml(cfg)
    # Parse config
    input_cfg = get(cfg, "input", Dict{String, Any}())
    binary_paths = [expanduser(String(p)) for p in get(input_cfg, "binary_paths", String[])]
    isempty(binary_paths) && error("no binary_paths in [input]")

    run_cfg = get(cfg, "run", Dict{String, Any}())
    scheme_name = Symbol(lowercase(String(get(run_cfg, "scheme", "upwind"))))
    Hp = Int(get(run_cfg, "halo_padding", scheme_name in (:ppm, :linrood) ? 3 :
                                          scheme_name == :slopes ? 2 : 1))
    ppm_order = Int(get(run_cfg, "ppm_order", 5))
    start_window = Int(get(run_cfg, "start_window", 1))
    stop_window_override = get(run_cfg, "stop_window", nothing)

    output_cfg = get(cfg, "output", Dict{String, Any}())
    snapshot_hours = Float64.(get(output_cfg, "snapshot_hours", [0, 6, 12, 18, 24, 30, 36, 42, 48]))
    snapshot_file = expanduser(String(get(output_cfg, "snapshot_file", "cs_snapshot.nc")))

    # Parse tracers
    tracers_cfg = get(cfg, "tracers", Dict{String, Any}())
    tracer_names = Symbol[]
    tracer_inits = Dict{Symbol, Any}()
    if isempty(tracers_cfg)
        push!(tracer_names, :CO2)
        tracer_inits[:CO2] = Dict("kind" => "uniform", "background" => 4.11e-4)
    else
        for (name, tcfg) in tracers_cfg
            sym = Symbol(name)
            push!(tracer_names, sym)
            tracer_inits[sym] = get(tcfg, "init", Dict("kind" => "uniform", "background" => 0.0))
        end
    end

    # Open first binary
    reader1 = CubedSphereBinaryReader(first(binary_paths); FT=Float64)
    h = reader1.header
    Nc = h.Nc; Nz = h.nlevel
    steps_per_window = h.steps_per_window
    window_hours = h.dt_met_seconds / 3600.0

    println("  Binary convention: $(h.panel_convention)")
    mesh = CubedSphereMesh(; Nc=Nc, Hp=Hp, convention=mesh_convention(reader1))
    scheme = if scheme_name == :slopes
        SlopesScheme()
    elseif scheme_name == :ppm
        PPMScheme()
    else
        UpwindScheme()
    end

    println("="^72)
    println("CS transport runner")
    println("  Grid: C$Nc, $Nz levels, Hp=$Hp")
    println("  Scheme: $scheme_name" * (scheme_name in (:ppm, :linrood) ? " (ord=$ppm_order)" : ""))
    println("  Tracers: $(join(String.(tracer_names), ", "))")
    println("  Binaries: $(length(binary_paths))")
    println("  Output: $snapshot_file")
    println("  Snapshots at hours: $snapshot_hours")
    println("="^72)

    # Precompute CS cell center lat-lon
    cs_lons = [panel_cell_center_lonlat(Nc, p, Float64)[1] for p in 1:6]
    cs_lats = [panel_cell_center_lonlat(Nc, p, Float64)[2] for p in 1:6]

    # Initialize air mass from window 1
    pm_raw, _, _, _, _ = load_cs_window(reader1, 1)
    pm = ntuple(p -> pad_panel(pm_raw[p], Hp), 6)
    fill_panel_halos!(pm, mesh; dir=1)

    # Initialize tracers
    tracer_panels = Dict{Symbol, NTuple{6, Array{Float64, 3}}}()
    for name in tracer_names
        init_cfg = tracer_inits[name]
        kind = Symbol(lowercase(String(get(init_cfg, "kind", "uniform"))))
        if kind == :uniform
            bg = Float64(get(init_cfg, "background", 0.0))
            # rm = q * m
            tracer_panels[name] = ntuple(p -> pm[p] .* bg, 6)
        elseif kind == :catrine_co2
            bg = 4.11e-4  # ~411 ppm as mixing ratio
            tracer_panels[name] = ntuple(p -> pm[p] .* bg, 6)
        else
            tracer_panels[name] = ntuple(p -> zeros(Float64, size(pm[p])), 6)
        end
        fill_panel_halos!(tracer_panels[name], mesh; dir=1)
    end

    # Workspaces
    ws = CSAdvectionWorkspace(mesh, Nz)
    ws_lr = scheme_name == :linrood ? LinRoodWorkspace(mesh; FT=Float64, Nz=Nz) : nothing

    # Snapshot storage
    snap_data = Dict{Symbol, NTuple{6, Array{Float64, 3}}}[]  # per-tracer rm
    snap_m    = NTuple{6, Array{Float64, 3}}[]                 # air mass

    function _capture!()
        push!(snap_m, ntuple(p -> copy(pm[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]), 6))
        d = Dict{Symbol, NTuple{6, Array{Float64, 3}}}()
        for name in tracer_names
            d[name] = ntuple(p -> copy(tracer_panels[name][p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]), 6)
        end
        push!(snap_data, d)
    end

    snap_idx = 1
    total_hour = 0.0

    # Initial snapshot
    if snap_idx <= length(snapshot_hours) && abs(snapshot_hours[snap_idx]) < 0.5
        _capture!()
        @printf("  Snapshot %d at t=%.0fh\n", snap_idx, 0.0)
        snap_idx += 1
    end

    # Open all readers
    readers = [reader1]
    for path in binary_paths[2:end]
        push!(readers, CubedSphereBinaryReader(expanduser(path); FT=Float64))
    end

    t0 = time()

    for reader in readers
        n_win = reader.header.nwindow
        stop_win = stop_window_override === nothing ? n_win : min(Int(stop_window_override), n_win)

        for win in start_window:stop_win
            pm_w, _, pam_w, pbm_w, pcm_w = load_cs_window(reader, win)

            # Update air mass from binary
            for p in 1:6
                pm[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, :] .= pm_w[p]
            end
            fill_panel_halos!(pm, mesh; dir=1)

            # Pad fluxes to haloed size
            # strang_split_cs! handles CFL subcycling internally (raw window fluxes OK)
            # LinRood has no internal CFL pilot — compute max CFL and subdivide here
            if scheme_name == :linrood
                # Compute per-substep CFL to determine number of LinRood sub-passes
                max_cfl_lr = 0.0
                for p in 1:6
                    local Nc_f = size(pam_w[p], 1) - 1
                    for k in axes(pam_w[p], 3), j in 1:Nc_f, i in 1:Nc_f+1
                        f = abs(pam_w[p][i, j, k]) / steps_per_window
                        donor = (i >= 2 && i <= Nc_f) ? pm_raw[p][min(i, Nc_f), j, k] : pm_raw[p][max(1, i-1), j, k]
                        donor > 0 && (max_cfl_lr = max(max_cfl_lr, f / donor))
                    end
                    local Nc_g = size(pbm_w[p], 2) - 1
                    for k in axes(pbm_w[p], 3), j in 1:Nc_g+1, i in 1:Nc_g
                        f = abs(pbm_w[p][i, j, k]) / steps_per_window
                        donor = (j >= 2 && j <= Nc_g) ? pm_raw[p][i, min(j, Nc_g), k] : pm_raw[p][i, max(1, j-1), k]
                        donor > 0 && (max_cfl_lr = max(max_cfl_lr, f / donor))
                    end
                end
                n_lr_sub = max(1, ceil(Int, max_cfl_lr / 0.85))
                total_sub = steps_per_window * n_lr_sub
                fs = 1.0 / total_sub
                @printf("  Window %d: CFL/substep=%.1f → %d LinRood sub-passes/substep (total %d)\n",
                        win, max_cfl_lr, n_lr_sub, total_sub)
                pam_p = ntuple(p -> pad_am(pam_w[p] .* fs, Hp), 6)
                pbm_p = ntuple(p -> pad_bm(pbm_w[p] .* fs, Hp), 6)
                pcm_p = ntuple(p -> pad_cm(pcm_w[p] .* fs, Hp), 6)
            else
                n_lr_sub = 1
                pam_p = ntuple(p -> pad_am(pam_w[p], Hp), 6)
                pbm_p = ntuple(p -> pad_bm(pbm_w[p], Hp), 6)
                pcm_p = ntuple(p -> pad_cm(pcm_w[p], Hp), 6)
            end

            # Substep loop
            for sub in 1:steps_per_window
                for lr_sub in 1:n_lr_sub
                    # Save air mass before first tracer — each tracer must see
                    # the same starting mass.  strang_split_cs! updates pm in
                    # place, so without this restore the mass would evolve
                    # N_tracers times per substep instead of once.
                    pm_snap = ntuple(p -> copy(pm[p]), 6)
                    for name in tracer_names
                        for p in 1:6
                            pm[p] .= pm_snap[p]
                        end
                        if scheme_name == :linrood
                            strang_split_linrood_ppm!(
                                tracer_panels[name], pm, pam_p, pbm_p, pcm_p,
                                mesh, Val(ppm_order), ws, ws_lr)
                        else
                            strang_split_cs!(tracer_panels[name], pm, pam_p, pbm_p, pcm_p,
                                              mesh, scheme, ws; cfl_limit=0.95)
                        end
                    end
                end
            end

            total_hour += window_hours

            # Check snapshots
            while snap_idx <= length(snapshot_hours) &&
                  abs(total_hour - snapshot_hours[snap_idx]) < 0.5
                _capture!()
                @printf("  Snapshot %d at t=%.0fh\n", snap_idx, total_hour)
                snap_idx += 1
            end
        end
        # Reset start_window for subsequent binaries
        start_window = 1
    end

    elapsed = time() - t0
    @printf("\nRun complete: %.1fs (%d windows)\n", elapsed, length(snap_data))

    export_snapshot(snapshot_file, cs_lons, cs_lats, snap_data, snap_m,
                    snapshot_hours, Nc, Nz)

    # Print mass diagnostics
    for name in tracer_names
        rm = tracer_panels[name]
        total = sum(sum(rm[p][Hp+1:Hp+Nc, Hp+1:Hp+Nc, :]) for p in 1:6)
        @printf("  Final tracer mass for %s: %.12e kg\n", String(name), total)
    end
end

function main()
    isempty(ARGS) && error("Usage: julia --project=. scripts/run_cs_transport.jl config.toml\n" *
                           "   or: julia --project=. scripts/run_cs_transport.jl --binary <path>")

    if endswith(ARGS[1], ".toml")
        cfg = TOML.parsefile(expanduser(ARGS[1]))
        run_cs_toml(cfg)
    else
        # Legacy command-line mode
        args = parse_args()
        binary_path = expanduser(get(args, "binary", ""))
        binary2_path = get(args, "day2", "")
        output_path = expanduser(get(args, "output", "cs_snapshot.nc"))
        hours_str = get(args, "hours", "0,6,12,24")
        snapshot_hours = [parse(Float64, s) for s in split(hours_str, ",")]

        isempty(binary_path) && error("--binary required")

        binary_paths = [binary_path]
        !isempty(binary2_path) && push!(binary_paths, expanduser(binary2_path))

        cfg = Dict{String, Any}(
            "input" => Dict("binary_paths" => binary_paths),
            "run"   => Dict("scheme" => "upwind"),
            "output" => Dict("snapshot_hours" => snapshot_hours,
                             "snapshot_file" => output_path),
        )
        run_cs_toml(cfg)
    end
end

main()
