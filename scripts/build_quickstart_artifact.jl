#!/usr/bin/env julia
# ===========================================================================
# Build quickstart artifact: download GEOS-FP via OPeNDAP → compute mass
# fluxes → write binary for PreprocessedLatLonMetDriver.
#
# Usage (maintainer):
#   julia --project=. scripts/build_quickstart_artifact.jl [output_dir]
#
# Also callable from quickstart.jl:
#   include("build_quickstart_artifact.jl")
#   build_quickstart_data("/path/to/output")
#
# No authentication required (GEOS-FP OPeNDAP is public).
# Output: ~50-100 MB of preprocessed binary mass fluxes for a 7-day run.
# ===========================================================================

using NCDatasets
using JSON3
using Printf
using Dates

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

const OPENDAP_URL = "https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/assim/inst3_3d_asm_Nv"
const DATE_START  = DateTime(2025, 2, 1, 0, 0, 0)
const DATE_END    = DateTime(2025, 2, 7, 21, 0, 0)
const LON_STRIDE  = 10   # ~3.75° lon spacing → ~96 points
const LAT_STRIDE  = 10   # ~2.5° lat spacing  → ~73 points
const HEADER_SIZE = 4096
const FT          = Float32
const R_EARTH     = 6.371e6  # meters
const G           = 9.80665  # m/s²

# ---------------------------------------------------------------------------
# Step 1: Download coarsened GEOS-FP data via OPeNDAP
# ---------------------------------------------------------------------------

function download_geosfp_data(outdir::String)
    nc_path = joinpath(outdir, "geosfp_raw.nc")
    if isfile(nc_path) && filesize(nc_path) > 1000
        @info "Raw data already cached: $nc_path"
        return nc_path
    end

    mkpath(outdir)
    @info "Connecting to GEOS-FP OPeNDAP: $OPENDAP_URL"
    ds = NCDataset(OPENDAP_URL)

    times_all = ds["time"][:]
    lon_full  = ds["lon"][:]
    lat_full  = ds["lat"][:]

    lon_idx = 1:LON_STRIDE:length(lon_full)
    lat_idx = 1:LAT_STRIDE:length(lat_full)
    lon_sub = lon_full[lon_idx]
    lat_sub = lat_full[lat_idx]
    Nx = length(lon_sub)
    Ny = length(lat_sub)
    Nz = length(ds["lev"][:])

    tidx_start = findfirst(t -> t >= DATE_START, times_all)
    tidx_end   = findlast(t -> t <= DATE_END, times_all)
    (tidx_start !== nothing && tidx_end !== nothing) ||
        error("Time range not found in OPeNDAP dataset")
    time_indices = tidx_start:tidx_end
    Nt = length(time_indices)

    @info "Grid: $Nx lon × $Ny lat × $Nz lev × $Nt timesteps"

    NCDataset(nc_path, "c") do out
        defDim(out, "lon", Nx)
        defDim(out, "lat", Ny)
        defDim(out, "lev", Nz)
        defDim(out, "time", Nt)

        defVar(out, "lon", Float64, ("lon",))[:] = lon_sub
        defVar(out, "lat", Float64, ("lat",))[:] = lat_sub
        defVar(out, "time", Float64, ("time",);
               attrib = Dict("units" => "hours since 2000-01-01"))

        for vname in ["u", "v", "delp", "ps"]
            dims = vname == "ps" ? ("lon", "lat", "time") : ("lon", "lat", "lev", "time")
            defVar(out, vname, Float32, dims; fillvalue = Float32(-9999))
        end

        out.attrib["source"] = "GEOS-FP via OPeNDAP"
        out.attrib["url"] = OPENDAP_URL

        for (local_t, global_t) in enumerate(time_indices)
            t_str = string(times_all[global_t])
            print("  [$local_t/$Nt] $t_str ... ")
            t0 = time()

            for attempt in 1:3
                try
                    out["u"][:, :, :, local_t]  = Float32.(ds["u"][lon_idx, lat_idx, :, global_t])
                    out["v"][:, :, :, local_t]  = Float32.(ds["v"][lon_idx, lat_idx, :, global_t])
                    out["delp"][:, :, :, local_t] = Float32.(ds["delp"][lon_idx, lat_idx, :, global_t])
                    out["ps"][:, :, local_t]    = Float32.(ds["ps"][lon_idx, lat_idx, global_t])
                    out["time"][local_t] = Dates.value(times_all[global_t] - DateTime(2000, 1, 1)) / 3600000.0
                    break
                catch e
                    attempt < 3 ? (println("retry $attempt..."); sleep(5)) : rethrow()
                end
            end
            println("done ($(round(time() - t0, digits=1))s)")
        end
    end

    close(ds)
    @info "Downloaded: $nc_path ($(round(filesize(nc_path) / 1e6, digits=1)) MB)"
    return nc_path
end

# ---------------------------------------------------------------------------
# Step 2: Compute mass fluxes and write binary
# ---------------------------------------------------------------------------

function compute_and_write_binary(nc_path::String, outdir::String)
    bin_path = joinpath(outdir, "massflux_geosfp_quickstart_float32.bin")
    if isfile(bin_path) && filesize(bin_path) > 1000
        @info "Binary already exists: $bin_path"
        return bin_path
    end

    @info "Computing mass fluxes from raw data..."
    ds = NCDataset(nc_path, "r")

    lons = Float64.(ds["lon"][:])
    lats = Float64.(ds["lat"][:])
    Nx = length(lons)
    Ny = length(lats)
    Nz = ds.dim["lev"]
    Nt = ds.dim["time"]

    # Grid geometry (lat-lon)
    Δlon = FT(abs(lons[2] - lons[1]))
    Δlat = FT(abs(lats[2] - lats[1]))

    # Cell areas and face lengths
    area_j = FT[R_EARTH^2 * deg2rad(Δlon) * abs(sind(lats[j] + Δlat / 2) - sind(lats[j] - Δlat / 2)) for j in 1:Ny]
    dy_j   = FT[R_EARTH * deg2rad(Δlat) for _ in 1:Ny]

    # dx at v-faces (includes boundaries at j=1 and j=Ny+1)
    dx_face = FT[R_EARTH * deg2rad(Δlon) * cosd(
        j == 1 ? lats[1] - Δlat / 2 :
        j == Ny + 1 ? lats[Ny] + Δlat / 2 :
        (lats[j - 1] + lats[j]) / 2
    ) for j in 1:Ny+1]

    met_interval = 10800.0  # 3-hourly
    dt = 1800.0             # advection sub-step
    half_dt = FT(dt / 2)
    steps_per_met = round(Int, met_interval / dt)

    # Sizes for staggered grids
    n_m  = Nx * Ny * Nz
    n_am = (Nx + 1) * Ny * Nz
    n_bm = Nx * (Ny + 1) * Nz
    n_cm = Nx * Ny * (Nz + 1)
    n_ps = Nx * Ny

    # Pre-allocate
    u_stag = Array{FT}(undef, Nx + 1, Ny, Nz)
    v_stag = Array{FT}(undef, Nx, Ny + 1, Nz)
    m      = Array{FT}(undef, Nx, Ny, Nz)
    am     = Array{FT}(undef, Nx + 1, Ny, Nz)
    bm     = Array{FT}(undef, Nx, Ny + 1, Nz)
    cm     = Array{FT}(undef, Nx, Ny, Nz + 1)

    # Write header
    header = Dict{String, Any}(
        "magic"                => "MFLX",
        "version"              => 1,
        "Nx"                   => Nx,
        "Ny"                   => Ny,
        "Nz"                   => Nz,
        "Nt"                   => Nt,
        "float_type"           => "Float32",
        "float_bytes"          => 4,
        "header_bytes"         => HEADER_SIZE,
        "n_m"                  => n_m,
        "n_am"                 => n_am,
        "n_bm"                 => n_bm,
        "n_cm"                 => n_cm,
        "n_ps"                 => n_ps,
        "dt_seconds"           => dt,
        "half_dt_seconds"      => dt / 2,
        "steps_per_met_window" => steps_per_met,
        "level_top"            => 1,
        "level_bot"            => Nz,
        "lons"                 => lons,
        "lats"                 => lats,
    )

    header_json = JSON3.write(header)
    length(header_json) < HEADER_SIZE ||
        error("Header too large ($(length(header_json)) >= $HEADER_SIZE)")

    open(bin_path, "w") do io
        header_buf = zeros(UInt8, HEADER_SIZE)
        copyto!(header_buf, 1, Vector{UInt8}(header_json), 1, length(header_json))
        write(io, header_buf)

        for win in 1:Nt
            t0 = time()

            # Read raw fields
            u_cc   = FT.(ds["u"][:, :, :, win])
            v_cc   = FT.(ds["v"][:, :, :, win])
            delp   = FT.(ds["delp"][:, :, :, win])
            ps_raw = FT.(ds["ps"][:, :, win])

            # Stagger winds to cell faces
            @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
                ip = i == Nx ? 1 : i + 1
                u_stag[i, j, k] = (u_cc[i, j, k] + u_cc[ip, j, k]) / 2
            end
            u_stag[Nx + 1, :, :] .= u_stag[1, :, :]

            @inbounds for k in 1:Nz, j in 2:Ny, i in 1:Nx
                v_stag[i, j, k] = (v_cc[i, j - 1, k] + v_cc[i, j, k]) / 2
            end
            v_stag[:, 1, :] .= 0
            v_stag[:, Ny + 1, :] .= 0

            # Air mass: m = delp * area / g
            @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
                m[i, j, k] = delp[i, j, k] * area_j[j] / FT(G)
            end

            # X-direction mass flux: am = u * Δp_avg * dy / g * half_dt
            @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx+1
                ip = i <= Nx ? i : 1
                im = i > 1 ? i - 1 : Nx
                dp_avg = (delp[im, j, k] + delp[ip, j, k]) / 2
                am[i, j, k] = u_stag[i, j, k] * dp_avg * dy_j[j] / FT(G) * half_dt
            end

            # Y-direction mass flux: bm = v * Δp_avg * dx / g * half_dt
            @inbounds for k in 1:Nz, j in 1:Ny+1, i in 1:Nx
                if j == 1 || j == Ny + 1
                    bm[i, j, k] = zero(FT)
                else
                    dp_avg = (delp[i, j - 1, k] + delp[i, j, k]) / 2
                    bm[i, j, k] = v_stag[i, j, k] * dp_avg * dx_face[j] / FT(G) * half_dt
                end
            end

            # Vertical mass flux from continuity: cm(top)=0, integrate down
            fill!(cm, zero(FT))
            @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
                ip = i == Nx ? 1 : i + 1
                div_h = (am[ip, j, k] - am[i, j, k]) + (bm[i, j + 1, k] - bm[i, j, k])
                cm[i, j, k + 1] = cm[i, j, k] - div_h
            end

            write(io, vec(m))
            write(io, vec(am))
            write(io, vec(bm))
            write(io, vec(cm))
            write(io, vec(ps_raw))

            if win <= 2 || win == Nt || win % 10 == 0
                @info @sprintf("  Window %d/%d (%.1fs)", win, Nt, time() - t0)
            end
        end
    end

    close(ds)
    @info "Binary written: $bin_path ($(round(filesize(bin_path) / 1e6, digits=1)) MB)"
    return bin_path
end

# ---------------------------------------------------------------------------
# Step 3: Create tarball for hosting as artifact (maintainer only)
# ---------------------------------------------------------------------------

function create_tarball(outdir::String)
    bin_files = filter(f -> endswith(f, ".bin"), readdir(outdir; join=true))
    isempty(bin_files) && error("No .bin files found in $outdir")

    tarball = joinpath(dirname(outdir), "quickstart_met_data.tar.gz")
    @info "Creating tarball: $tarball"

    # Use tar relative to outdir so artifact unpacks cleanly
    run(`tar -czf $tarball -C $(dirname(outdir)) $(basename(outdir))`)

    # Compute SHA256
    sha = bytes2hex(open(io -> Pkg.Types.sha2_256(io), tarball))
    sz = round(filesize(tarball) / 1e6, digits=1)
    @info "Tarball: $tarball ($sz MB)"
    @info "SHA256: $sha"
    println()
    println("Add to Artifacts.toml:")
    println("""
[quickstart_met_data]
git-tree-sha1 = "<run `julia -e 'using Pkg; Pkg.Artifacts.create_artifact(...)' to get this>"
lazy = true

    [[quickstart_met_data.download]]
    url = "https://github.com/RemoteSensingTools/AtmosTransport/releases/download/data-v1/quickstart_met_data.tar.gz"
    sha256 = "$sha"
""")

    return tarball
end

# ---------------------------------------------------------------------------
# Public entry point (called from quickstart.jl fallback)
# ---------------------------------------------------------------------------

function build_quickstart_data(outdir::String)
    nc_path = download_geosfp_data(outdir)
    bin_path = compute_and_write_binary(nc_path, outdir)
    @info "Quickstart data ready at: $outdir"
    return outdir
end

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    import Pkg
    outdir = length(ARGS) >= 1 ? ARGS[1] :
        joinpath(homedir(), "data", "metDrivers", "geosfp", "quickstart")
    build_quickstart_data(outdir)
    create_tarball(outdir)
end
