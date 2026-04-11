#!/usr/bin/env julia
#
# Standalone CS transport runner — runs CS advection on C90 binary
# and exports snapshot NetCDF files for visualization.
#
# This is a simplified version that doesn't depend on DrivenSimulation
# (which has src_v2 runtime complexities). It directly uses
# strang_split_cs! and exports snapshots on a regridded lat-lon grid
# for easy plotting.
#
# Usage:
#   julia --project=. scripts/run_cs_transport_simple.jl \
#       --binary <cs_binary.bin> --output <snapshot.nc> \
#       [--hours 0,6,12,24] [--day2 <binary2.bin>]

using Printf
using NCDatasets

include(joinpath(@__DIR__, "..", "src_v2", "AtmosTransportV2.jl"))
using .AtmosTransportV2
using .AtmosTransportV2.Operators.Advection: fill_panel_halos!, strang_split_cs!, CSAdvectionWorkspace

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

"""Panel cell centers in lat-lon degrees (for viz)."""
function panel_cell_center_lonlat(Nc::Int, p::Int, FT::Type{<:AbstractFloat})
    dα = FT(π) / (2 * Nc)
    α_centers = [FT(-π/4) + (i - 0.5) * dα for i in 1:Nc]
    lons = zeros(FT, Nc, Nc)
    lats = zeros(FT, Nc, Nc)
    for j in 1:Nc, i in 1:Nc
        ξ = tan(α_centers[i])
        η = tan(α_centers[j])
        d = one(FT) / sqrt(one(FT) + ξ^2 + η^2)
        x, y, z = if p == 1
            (d, ξ*d, η*d)
        elseif p == 2
            (-ξ*d, d, η*d)
        elseif p == 3
            (-d, -ξ*d, η*d)
        elseif p == 4
            (ξ*d, -d, η*d)
        elseif p == 5
            (-η*d, ξ*d, d)
        else
            (η*d, ξ*d, -d)
        end
        lons[i, j] = atand(y, x)
        lats[i, j] = asind(z / sqrt(x^2 + y^2 + z^2))
        lons[i, j] < 0 && (lons[i, j] += 360)
    end
    return lons, lats
end

function export_snapshot(nc_path::String, cs_lons, cs_lats, panels_rm, panels_m,
                         snapshot_hours::Vector{Float64}, tracer_name::String)
    # snapshot_data: (ntime, 6, Nc, Nc) arrays for column mean and surface
    Nc = size(panels_rm[1][1], 1)
    ntime = length(snapshot_hours)
    Nz = size(panels_rm[1][1], 3)

    # Compute column mean and surface (k=Nz) VMR
    # panels_rm, panels_m are Vector{NTuple{6}} indexed by time
    col_mean = zeros(Float64, 6 * Nc * Nc, ntime)
    surface  = zeros(Float64, 6 * Nc * Nc, ntime)
    lons_flat = zeros(Float64, 6 * Nc * Nc)
    lats_flat = zeros(Float64, 6 * Nc * Nc)

    idx = 1
    for p in 1:6, j in 1:Nc, i in 1:Nc
        lons_flat[idx] = cs_lons[p][i, j]
        lats_flat[idx] = cs_lats[p][i, j]
        for t in 1:ntime
            m_col = 0.0; rm_col = 0.0
            for k in 1:Nz
                m_col += panels_m[t][p][i, j, k]
                rm_col += panels_rm[t][p][i, j, k]
            end
            col_mean[idx, t] = m_col > 0 ? rm_col / m_col : 0.0
            m_sfc = panels_m[t][p][i, j, Nz]
            rm_sfc = panels_rm[t][p][i, j, Nz]
            surface[idx, t] = m_sfc > 0 ? rm_sfc / m_sfc : 0.0
        end
        idx += 1
    end

    isfile(nc_path) && rm(nc_path)
    ds = NCDataset(nc_path, "c")
    defDim(ds, "cell", 6 * Nc * Nc)
    defDim(ds, "time", ntime)
    ds.attrib["grid_type"] = "reduced_gaussian"  # use RG-style for viz compatibility
    ds.attrib["scheme"] = "cs_gamma_upwind"
    ds.attrib["tracer_name"] = tracer_name
    ds.attrib["Nc"] = Nc

    v_lon = defVar(ds, "lon_cell", Float64, ("cell",))
    v_lat = defVar(ds, "lat_cell", Float64, ("cell",))
    v_time = defVar(ds, "time_hours", Float64, ("time",))
    v_col = defVar(ds, "co2_column_mean", Float64, ("cell", "time"))
    v_sfc = defVar(ds, "co2_surface", Float64, ("cell", "time"))

    v_lon[:] = lons_flat
    v_lat[:] = lats_flat
    v_time[:] = snapshot_hours
    v_col[:, :] = col_mean
    v_sfc[:, :] = surface

    close(ds)
    println("Saved: $nc_path ($(6*Nc*Nc) cells × $ntime snapshots)")
end

function main()
    args = parse_args()
    binary_path = expanduser(get(args, "binary", ""))
    binary2_path = get(args, "day2", "")
    output_path = expanduser(get(args, "output", "cs_snapshot.nc"))
    hours_str = get(args, "hours", "0,6,12,24")
    snapshot_hours = [parse(Float64, s) for s in split(hours_str, ",")]

    isempty(binary_path) && error("--binary required")

    println("="^72)
    println("CS transport runner")
    println("  Binary: $binary_path")
    println("  Output: $output_path")
    println("  Snapshots at hours: $snapshot_hours")
    println("="^72)

    reader1 = CubedSphereBinaryReader(binary_path; FT=Float64)
    h = reader1.header
    Nc = h.Nc; Nz = h.nlevel; Hp = 1
    mesh = CubedSphereMesh(Nc=Nc, Hp=Hp)
    println("Grid: C$Nc, $Nz levels, $(h.nwindow) windows")

    # Precompute CS cell center lat-lon for export
    cs_lons = [panel_cell_center_lonlat(Nc, p, Float64)[1] for p in 1:6]
    cs_lats = [panel_cell_center_lonlat(Nc, p, Float64)[2] for p in 1:6]

    pad3(a) = let (i,j,k)=size(a); p=zeros(eltype(a),i+2,j+2,k); p[2:i+1,2:j+1,:].=a; p; end

    # Initialize state from window 1
    pm_raw, _, _, _, _ = load_cs_window(reader1, 1)
    pm  = ntuple(p -> pad3(pm_raw[p]), 6)
    prm = ntuple(p -> pad3(pm_raw[p] .* 4.11e-4), 6)  # 411 ppm uniform IC
    fill_panel_halos!(pm, mesh; dir=1)
    fill_panel_halos!(prm, mesh; dir=1)

    ws = CSAdvectionWorkspace(mesh, Nz)

    # Storage for snapshots
    n_snap = length(snapshot_hours)
    snap_m  = Vector{NTuple{6, Array{Float64, 3}}}(undef, n_snap)
    snap_rm = Vector{NTuple{6, Array{Float64, 3}}}(undef, n_snap)
    snap_idx = 1

    # Store initial state if 0.0 in hours
    if snapshot_hours[1] == 0.0
        snap_m[1]  = ntuple(p -> copy(pm[p][2:Nc+1, 2:Nc+1, :]), 6)
        snap_rm[1] = ntuple(p -> copy(prm[p][2:Nc+1, 2:Nc+1, :]), 6)
        snap_idx = 2
    end

    readers = [reader1]
    if !isempty(binary2_path)
        push!(readers, CubedSphereBinaryReader(expanduser(binary2_path); FT=Float64))
    end

    total_hour = 0.0
    t0 = time()

    for (reader_idx, reader) in enumerate(readers)
        n_win = reader.header.nwindow
        for win in 1:n_win
            pm_w, _, pam_w, pbm_w, pcm_w = load_cs_window(reader, win)

            # Reset air mass from binary (standard window carry-through pattern)
            for p in 1:6
                pm[p][2:Nc+1, 2:Nc+1, :] .= pm_w[p]
            end
            fill_panel_halos!(pm, mesh; dir=1)

            # Pad fluxes
            pam_p = ntuple(6) do p; z=zeros(Float64,Nc+3,Nc+2,Nz); z[2:Nc+2,2:Nc+1,:].=pam_w[p]; z end
            pbm_p = ntuple(6) do p; z=zeros(Float64,Nc+2,Nc+3,Nz); z[2:Nc+1,2:Nc+2,:].=pbm_w[p]; z end
            pcm_p = ntuple(6) do p; z=zeros(Float64,Nc+2,Nc+2,Nz+1); z[2:Nc+1,2:Nc+1,:].=pcm_w[p]; z end

            # 4 substeps per window
            for sub in 1:4
                strang_split_cs!(prm, pm, pam_p, pbm_p, pcm_p, mesh, UpwindScheme(), ws; cfl_limit=0.95)
            end

            total_hour += 1.0

            # Check if we need to snapshot
            while snap_idx <= n_snap && abs(snapshot_hours[snap_idx] - total_hour) < 0.5
                snap_m[snap_idx]  = ntuple(p -> copy(pm[p][2:Nc+1, 2:Nc+1, :]), 6)
                snap_rm[snap_idx] = ntuple(p -> copy(prm[p][2:Nc+1, 2:Nc+1, :]), 6)
                @printf("  Snapshot %d at t=%.0fh (global hour %.0f)\n",
                        snap_idx, snapshot_hours[snap_idx], total_hour)
                snap_idx += 1
            end
        end
    end

    # Fill any missing snapshots with the last state
    while snap_idx <= n_snap
        snap_m[snap_idx]  = ntuple(p -> copy(pm[p][2:Nc+1, 2:Nc+1, :]), 6)
        snap_rm[snap_idx] = ntuple(p -> copy(prm[p][2:Nc+1, 2:Nc+1, :]), 6)
        snap_idx += 1
    end

    elapsed = time() - t0
    println("\nRun complete: $(round(elapsed, digits=1))s")

    export_snapshot(output_path, cs_lons, cs_lats, snap_rm, snap_m, snapshot_hours, "co2")
end

main()
