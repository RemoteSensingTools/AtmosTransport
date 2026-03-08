#!/usr/bin/env julia
#
# Diagnostic script: check CATRINE initial conditions for anomalies
#
# Inspects the raw IC NetCDF files and reproduces the nearest-neighbor
# regridding to CS C180 to identify unphysical peaks (e.g., south of Australia).
#
# Usage:
#   julia --project=. scripts/diagnostics/check_catrine_ics.jl
#

using NCDatasets
using Printf
using Statistics
using Dates

# ── Configuration ──────────────────────────────────────────────────────────

const IC_FILES = [
    (species = :co2, variable = "CO2",
     file = expanduser("~/data/AtmosTransport/catrine/InitialConditions/startCO2_202112010000.nc")),
    (species = :sf6, variable = "SF6",
     file = expanduser("~/data/AtmosTransport/catrine/InitialConditions/startSF6_202112010000.nc")),
]

const Nc = 180  # C180 cubed sphere

# ── Gnomonic grid generation (self-contained, matches cubed_sphere_grid.jl) ─

function _gnomonic_xyz(ξ, η, panel::Int)
    d = 1.0 / sqrt(1.0 + ξ^2 + η^2)
    if     panel == 1; return ( d,  η*d,  -ξ*d)  # original has different signs but
    elseif panel == 2; return (-ξ*d,  d,  η*d)
    elseif panel == 3; return (-d, -ξ*d,  η*d)
    elseif panel == 4; return ( ξ*d, -d,  η*d)
    elseif panel == 5; return (-η*d,  ξ*d,  d)
    else;              return ( η*d,  ξ*d, -d)
    end
end

function _xyz_to_lonlat(x, y, z)
    lon = atand(y, x)
    lat = asind(z / sqrt(x^2 + y^2 + z^2))
    return (lon, lat)
end

function generate_cs_centers(Nc::Int)
    dα = π / (2Nc)
    α_faces = [-π/4 + (i-1)*dα for i in 1:(Nc+1)]
    α_centers = [(α_faces[i] + α_faces[i+1]) / 2 for i in 1:Nc]

    λᶜ = [zeros(Nc, Nc) for _ in 1:6]
    φᶜ = [zeros(Nc, Nc) for _ in 1:6]

    for p in 1:6, j in 1:Nc, i in 1:Nc
        ξc = tan(α_centers[i])
        ηc = tan(α_centers[j])
        xyz = _gnomonic_xyz(ξc, ηc, p)
        lon, lat = _xyz_to_lonlat(xyz...)
        λᶜ[p][i, j] = lon
        φᶜ[p][i, j] = lat
    end
    return λᶜ, φᶜ
end

# ── Step 1: Inspect raw IC files ──────────────────────────────────────────

function inspect_ic_file(filepath::String, varname::String, species::Symbol)
    println("\n", "="^80)
    println("  IC FILE: $filepath")
    println("  Variable: $varname  (species: $species)")
    println("="^80)

    if !isfile(filepath)
        println("  *** FILE NOT FOUND ***")
        return nothing
    end

    ds = NCDataset(filepath)

    # Print all variables
    println("\n  Variables in file:")
    for (name, var) in ds
        dims = join(["$(d)=$(s)" for (d,s) in zip(dimnames(var), size(var))], ", ")
        println("    $name  ($dims)")
        # Check for fill value attributes
        for attr in ["_FillValue", "missing_value"]
            if haskey(var.attrib, attr)
                println("      $attr = $(var.attrib[attr])")
            end
        end
    end

    # Coordinates
    lon_candidates = ["lon", "longitude", "x"]
    lat_candidates = ["lat", "latitude", "y"]
    lev_candidates = ["lev", "level", "plev", "z", "hybrid", "nhym"]

    lon_var = findfirst(k -> haskey(ds, k), lon_candidates)
    lat_var = findfirst(k -> haskey(ds, k), lat_candidates)
    lev_var = findfirst(k -> haskey(ds, k), lev_candidates)

    lon_var = isnothing(lon_var) ? nothing : lon_candidates[lon_var]
    lat_var = isnothing(lat_var) ? nothing : lat_candidates[lat_var]
    lev_var = isnothing(lev_var) ? nothing : lev_candidates[lev_var]

    if isnothing(lon_var) || isnothing(lat_var) || isnothing(lev_var)
        println("  *** Could not find lon/lat/lev coordinates ***")
        println("  lon_var=$lon_var, lat_var=$lat_var, lev_var=$lev_var")
        close(ds)
        return nothing
    end

    lon_src = Float64.(ds[lon_var][:])
    lat_src = Float64.(ds[lat_var][:])
    lev_src = Float64.(ds[lev_var][:])

    println("\n  Coordinates:")
    println("    lon ($lon_var): $(length(lon_src)) values, range [$(minimum(lon_src)), $(maximum(lon_src))]")
    println("    lat ($lat_var): $(length(lat_src)) values, range [$(minimum(lat_src)), $(maximum(lat_src))]")
    println("    lev ($lev_var): $(length(lev_src)) values, range [$(minimum(lev_src)), $(maximum(lev_src))]")
    Δlon = length(lon_src) > 1 ? lon_src[2] - lon_src[1] : NaN
    Δlat = length(lat_src) > 1 ? lat_src[2] - lat_src[1] : NaN
    println("    Δlon = $Δlon, Δlat = $Δlat")
    println("    Lat order: $(lat_src[1] > lat_src[end] ? "N→S" : "S→N")")
    println("    Lon range: $(minimum(lon_src) < 0 ? "-180:180" : "0:360")")

    # Check ap/bp/Psurf
    has_hybrid = haskey(ds, "ap") && haskey(ds, "bp") && haskey(ds, "Psurf")
    println("\n  Hybrid-sigma coordinates: $has_hybrid")
    if has_hybrid
        ap = Float64.(ds["ap"][:])
        bp = Float64.(ds["bp"][:])
        println("    ap: length=$(length(ap)), range [$(minimum(ap)), $(maximum(ap))]")
        println("    bp: length=$(length(bp)), range [$(minimum(bp)), $(maximum(bp))]")
        println("    Expected Nlev+1 = $(length(lev_src)+1), got $(length(ap))")
        if length(ap) != length(lev_src) + 1
            println("    *** WARNING: ap/bp length mismatch! ***")
        end

        psurf_var = ds["Psurf"]
        psurf_raw_nc = psurf_var[:, :]
        # Check for missing values
        n_miss = count(ismissing, psurf_raw_nc)
        psurf_vals = collect(skipmissing(psurf_raw_nc))
        println("    Psurf: $(size(psurf_raw_nc)), missing=$n_miss, " *
                "min=$(minimum(psurf_vals)), max=$(maximum(psurf_vals)), mean=$(mean(psurf_vals))")
        if n_miss > 0
            println("    *** WARNING: $n_miss missing Psurf values (will default to 101325 Pa) ***")
        end
        n_zero = count(==(0), psurf_vals)
        if n_zero > 0
            println("    *** WARNING: $n_zero zero Psurf values ***")
        end
    end

    # Read tracer data
    if !haskey(ds, varname)
        println("  *** Variable '$varname' not found in file ***")
        close(ds)
        return nothing
    end

    raw_var = ds[varname]
    ndim = ndims(raw_var)
    println("\n  Tracer data '$varname': $(ndim)D, size=$(size(raw_var))")

    raw_nc = if ndim == 4
        raw_var[:, :, :, 1]
    elseif ndim == 3
        raw_var[:, :, :]
    else
        println("  *** Unexpected dimensionality $ndim ***")
        close(ds)
        return nothing
    end

    # Convert missing
    n_miss_raw = count(ismissing, raw_nc)
    println("  Missing values in raw data: $n_miss_raw")

    raw = Float64.(collect(replace(raw_nc, missing => NaN)))

    # Check for anomalous values
    n_nan = count(isnan, raw)
    n_inf = count(isinf, raw)
    n_fill_1e20 = count(x -> !isnan(x) && abs(x) > 1e15, raw)
    println("  NaN count: $n_nan")
    println("  Inf count: $n_inf")
    println("  |value| > 1e15 (potential fill): $n_fill_1e20")

    finite_vals = filter(isfinite, raw)
    if !isempty(finite_vals)
        println("  Finite values: min=$(minimum(finite_vals)), max=$(maximum(finite_vals)), " *
                "mean=$(mean(finite_vals)), std=$(std(finite_vals))")
    end

    # Per-level stats
    Nlev = size(raw, 3)
    println("\n  Per-level statistics (surface=level 1 or last):")
    levels_to_show = Nlev <= 10 ? (1:Nlev) : vcat(1:3, Nlev-2:Nlev)
    for k in levels_to_show
        slice = raw[:, :, k]
        fin = filter(isfinite, slice)
        if !isempty(fin)
            @printf("    Level %3d: min=%.6e  max=%.6e  mean=%.6e  nan=%d\n",
                    k, minimum(fin), maximum(fin), mean(fin), count(isnan, slice))
        end
    end

    # Flag extreme values — find locations
    global_mean = mean(finite_vals)
    global_std = std(finite_vals)
    threshold = global_mean + 5 * global_std
    if global_mean > 0
        println("\n  Extreme value threshold (mean + 5σ): $threshold")
        extremes = []
        for k in 1:size(raw, 3), j in 1:size(raw, 2), i in 1:size(raw, 1)
            v = raw[i, j, k]
            if isfinite(v) && v > threshold
                push!(extremes, (val=v, i=i, j=j, k=k,
                                  lon=lon_src[i], lat=lat_src[j]))
            end
        end
        if !isempty(extremes)
            sort!(extremes, by=x -> -x.val)
            n_show = min(10, length(extremes))
            println("  Found $(length(extremes)) extreme values, top $n_show:")
            for e in extremes[1:n_show]
                @printf("    [%d,%d,%d] lon=%.1f lat=%.1f val=%.6e\n",
                        e.i, e.j, e.k, e.lon, e.lat, e.val)
            end
        else
            println("  No extreme values found (all within 5σ of mean)")
        end
    end

    close(ds)
    return (lon_src=lon_src, lat_src=lat_src, lev_src=lev_src, raw=raw, has_hybrid=has_hybrid)
end

# ── Step 2: Reproduce IC regridding to CS C180 ───────────────────────────

function check_regridding(ic_data, species::Symbol)
    isnothing(ic_data) && return

    println("\n", "-"^80)
    println("  Regridding check for $species → CS C$Nc")
    println("-"^80)

    lon_src = ic_data.lon_src
    lat_src = ic_data.lat_src
    raw = ic_data.raw
    Nlev = size(raw, 3)

    # Apply the same transforms as initial_conditions.jl:
    # 1. Latitude S→N
    lat_use = copy(lat_src)
    raw_use = copy(raw)
    if length(lat_src) > 1 && lat_src[1] > lat_src[end]
        lat_use = reverse(lat_src)
        raw_use = raw_use[:, end:-1:1, :]
        println("  Reversed latitude (was N→S)")
    end

    # 2. Longitude → 0:360
    lon_use = copy(lon_src)
    if minimum(lon_src) < 0
        n = length(lon_src)
        split = findfirst(>=(0), lon_src)
        if split !== nothing
            idx = vcat(split:n, 1:split-1)
            lon_use = mod.(lon_src[idx], 360.0)
            raw_use = raw_use[idx, :, :]
            println("  Reordered longitude to 0:360 (split at index $split)")
        end
    end

    # 3. Vertical: check if flip needed
    lev_src = ic_data.lev_src
    if length(lev_src) > 1 && lev_src[1] > lev_src[end]
        raw_use = raw_use[:, :, end:-1:1]
        println("  Reversed vertical (was decreasing)")
    end

    Δlon = lon_use[2] - lon_use[1]
    Δlat = lat_use[2] - lat_use[1]
    Nlon_s = length(lon_use)
    Nlat_s = length(lat_use)

    println("  Source grid: $(Nlon_s) × $(Nlat_s) × $Nlev")
    println("  Δlon=$Δlon, Δlat=$Δlat")
    println("  lon_use range: [$(lon_use[1]), $(lon_use[end])]")
    println("  lat_use range: [$(lat_use[1]), $(lat_use[end])]")

    # Generate CS grid
    println("  Generating CS C$Nc grid coordinates...")
    λᶜ, φᶜ = generate_cs_centers(Nc)

    # Pick surface level (last level after potential flip = bottom of atmosphere)
    k_sfc = Nlev  # bottom level

    # Regrid surface level using exact same formula as initial_conditions.jl
    regridded = [zeros(Nc, Nc) for _ in 1:6]
    source_ii = [zeros(Int, Nc, Nc) for _ in 1:6]  # track which source cell was used
    source_jj = [zeros(Int, Nc, Nc) for _ in 1:6]
    mapping_error_lon = [zeros(Nc, Nc) for _ in 1:6]  # lon error in degrees
    mapping_error_lat = [zeros(Nc, Nc) for _ in 1:6]

    for p in 1:6, j in 1:Nc, i in 1:Nc
        lon = mod(λᶜ[p][i, j], 360.0)
        lat = φᶜ[p][i, j]

        ii = clamp(round(Int, (lon - lon_use[1]) / Δlon) + 1, 1, Nlon_s)
        jj = clamp(round(Int, (lat - lat_use[1]) / Δlat) + 1, 1, Nlat_s)

        regridded[p][i, j] = raw_use[ii, jj, k_sfc]
        source_ii[p][i, j] = ii
        source_jj[p][i, j] = jj

        # Check mapping accuracy
        mapped_lon = lon_use[ii]
        mapped_lat = lat_use[jj]
        # Handle 360° wrap for lon error
        dlon = abs(lon - mapped_lon)
        dlon = min(dlon, 360.0 - dlon)
        dlat = abs(lat - mapped_lat)
        mapping_error_lon[p][i, j] = dlon
        mapping_error_lat[p][i, j] = dlat
    end

    # Per-panel statistics
    println("\n  Per-panel surface-level statistics:")
    println("  Panel |    min        max        mean     | max_Δlon  max_Δlat  | idx range")
    println("  ------+--------------------------------------+---------------------+-----------")
    for p in 1:6
        vals = regridded[p]
        max_dlon = maximum(mapping_error_lon[p])
        max_dlat = maximum(mapping_error_lat[p])
        ii_range = extrema(source_ii[p])
        jj_range = extrema(source_jj[p])
        @printf("    %d   | %.4e  %.4e  %.4e | %6.2f°   %6.2f°   | ii=%s jj=%s\n",
                p, minimum(vals), maximum(vals), mean(vals),
                max_dlon, max_dlat,
                "$(ii_range[1])-$(ii_range[2])", "$(jj_range[1])-$(jj_range[2])")
    end

    # Check for NaN/Inf in regridded
    for p in 1:6
        n_nan = count(isnan, regridded[p])
        n_inf = count(isinf, regridded[p])
        if n_nan > 0 || n_inf > 0
            println("  *** Panel $p: $n_nan NaN, $n_inf Inf in regridded data ***")
        end
    end

    # Find global extreme cells
    all_vals = vcat([vec(regridded[p]) for p in 1:6]...)
    fin_vals = filter(isfinite, all_vals)
    global_mean = mean(fin_vals)
    global_std = std(fin_vals)

    println("\n  Global regridded stats: mean=$(global_mean), std=$(global_std)")

    # Top 20 extreme cells
    extremes = []
    for p in 1:6, j in 1:Nc, i in 1:Nc
        v = regridded[p][i, j]
        push!(extremes, (val=v, p=p, i=i, j=j,
                          lon=λᶜ[p][i,j], lat=φᶜ[p][i,j],
                          src_ii=source_ii[p][i,j], src_jj=source_jj[p][i,j],
                          dlon=mapping_error_lon[p][i,j],
                          dlat=mapping_error_lat[p][i,j]))
    end
    sort!(extremes, by=x -> isnan(x.val) ? -Inf : -x.val)

    println("\n  Top 20 highest values after regridding:")
    println("  Rank | Panel (i,j)  | CS lon/lat        | src (ii,jj) | Δlon   Δlat  | value")
    println("  -----+--------------+-------------------+-------------+--------------+--------")
    for (rank, e) in enumerate(extremes[1:min(20, length(extremes))])
        @printf("  %3d  |  %d  (%3d,%3d) | %7.2f° %7.2f° | (%3d,%3d)   | %5.2f° %5.2f° | %.6e\n",
                rank, e.p, e.i, e.j, e.lon, e.lat,
                e.src_ii, e.src_jj, e.dlon, e.dlat, e.val)
    end

    # Bottom 20 (in case of negative anomalies)
    sort!(extremes, by=x -> isnan(x.val) ? Inf : x.val)
    println("\n  Bottom 10 lowest values:")
    for (rank, e) in enumerate(extremes[1:min(10, length(extremes))])
        @printf("  %3d  |  %d  (%3d,%3d) | %7.2f° %7.2f° | (%3d,%3d)   | %5.2f° %5.2f° | %.6e\n",
                rank, e.p, e.i, e.j, e.lon, e.lat,
                e.src_ii, e.src_jj, e.dlon, e.dlat, e.val)
    end

    # Check for cells where mapping error is suspiciously large
    bad_mapping = [(p=e.p, i=e.i, j=e.j, lon=e.lon, lat=e.lat,
                    dlon=e.dlon, dlat=e.dlat, val=e.val)
                   for e in extremes
                   if e.dlon > 2*abs(Δlon) || e.dlat > 2*abs(Δlat)]
    if !isempty(bad_mapping)
        println("\n  *** WARNING: $(length(bad_mapping)) cells with mapping error > 2×Δ ***")
        for (idx, b) in enumerate(bad_mapping[1:min(20, length(bad_mapping))])
            @printf("    Panel %d (%3d,%3d): CS=(%.2f°,%.2f°) err=(%.2f°,%.2f°) val=%.6e\n",
                    b.p, b.i, b.j, b.lon, b.lat, b.dlon, b.dlat, b.val)
        end
    else
        println("\n  All cells have mapping error ≤ 2×Δ — nearest-neighbor looks correct")
    end

    # Check cells near Australia specifically (-10° to -60° lat, 100° to 180° lon)
    println("\n  Cells in Australia/Southern Ocean region (lat -60:-10, lon 100:180):")
    australia = [(p=e.p, i=e.i, j=e.j, lon=e.lon, lat=e.lat, val=e.val)
                 for e in extremes
                 if -60 <= e.lat <= -10 && 100 <= mod(e.lon, 360) <= 180]
    if !isempty(australia)
        sort!(australia, by=x -> -x.val)
        for a in australia[1:min(10, length(australia))]
            @printf("    Panel %d (%3d,%3d): (%.2f°,%.2f°) val=%.6e\n",
                    a.p, a.i, a.j, a.lon, a.lat, a.val)
        end
    else
        println("    No cells found in this region")
    end

    # Check uniqueness of source indices — are many CS cells mapping to same source?
    println("\n  Source cell usage (how many CS cells map to same source cell):")
    usage = Dict{Tuple{Int,Int}, Int}()
    for p in 1:6, j in 1:Nc, i in 1:Nc
        key = (source_ii[p][i,j], source_jj[p][i,j])
        usage[key] = get(usage, key, 0) + 1
    end
    max_usage = maximum(values(usage))
    println("    Max CS cells sharing one source cell: $max_usage")
    if max_usage > 10
        top_shared = sort(collect(usage), by=x -> -x[2])[1:min(10, length(usage))]
        println("    Top shared source cells:")
        for ((ii, jj), count) in top_shared
            @printf("      source (%3d,%3d) → lon=%.1f lat=%.1f : %d CS cells\n",
                    ii, jj, lon_use[ii], lat_use[jj], count)
        end
    end

    return regridded
end

# ── Step 3: Check preprocessed binary files ──────────────────────────────

function check_preprocessed_binaries()
    println("\n", "="^80)
    println("  Preprocessed binary file check")
    println("="^80)

    bin_dirs = [
        "/temp1/catrine/met/geosit_c180/massflux",
        "/temp1/catrine/met/geosit_c180/surface_bin",
        "/temp2/catrine-runs/met/geosit_c180",
    ]

    for dir in bin_dirs
        if !isdir(dir)
            println("  $dir — NOT FOUND")
            continue
        end

        files = readdir(dir, join=true)
        bin_files = filter(f -> endswith(f, ".bin"), files)
        println("\n  $dir: $(length(bin_files)) .bin files")

        if isempty(bin_files)
            println("    (no binary files)")
            continue
        end

        # Check file sizes for consistency
        sizes = [filesize(f) for f in bin_files]
        unique_sizes = sort(unique(sizes))
        println("    Unique file sizes: $(length(unique_sizes))")
        for s in unique_sizes
            n = count(==(s), sizes)
            @printf("      %12d bytes (%6.1f MB) × %d files\n", s, s/1e6, n)
        end

        # Check for suspiciously small files (corrupted?)
        small = [(basename(f), filesize(f)) for f in bin_files if filesize(f) < 1000]
        if !isempty(small)
            println("    *** WARNING: $(length(small)) suspiciously small files (<1KB):")
            for (name, sz) in small
                println("      $name: $sz bytes")
            end
        end

        # Check modification times
        mtimes = [mtime(f) for f in bin_files]
        oldest = minimum(mtimes)
        newest = maximum(mtimes)
        println("    Date range: $(Dates.unix2datetime(oldest)) → $(Dates.unix2datetime(newest))")
    end
end

# ── Main ─────────────────────────────────────────────────────────────────

function main()
    println("CATRINE IC Diagnostic Script")
    println("============================")
    println("CS resolution: C$Nc")
    println()

    # Step 1: Inspect raw IC files
    ic_results = Dict{Symbol, Any}()
    for ic in IC_FILES
        result = inspect_ic_file(ic.file, ic.variable, ic.species)
        ic_results[ic.species] = result
    end

    # Step 2: Regridding check
    for ic in IC_FILES
        result = ic_results[ic.species]
        if !isnothing(result)
            check_regridding(result, ic.species)
        end
    end

    # Step 3: Check preprocessed binaries
    check_preprocessed_binaries()

    println("\n", "="^80)
    println("  DIAGNOSTIC COMPLETE")
    println("="^80)
end

main()
