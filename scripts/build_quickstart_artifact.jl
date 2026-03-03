#!/usr/bin/env julia
# ===========================================================================
# Build quickstart artifact: download GEOS-IT C180 from WashU → preprocess
# to flat binary → create tarball for GitHub Release hosting.
#
# Usage (maintainer only):
#   julia --project=. scripts/build_quickstart_artifact.jl [output_dir]
#
# Steps:
#   1. Download 1 day of GEOS-IT C180 CTM_A1 from WashU archive (~4.2 GB)
#   2. Preprocess to flat binary via preprocess_geosfp_cs.jl (~4 GB)
#   3. Package preprocessed binary as tarball (~700 MB compressed)
#   4. Print SHA256 and Artifacts.toml entry
#
# No authentication required (WashU archive is public HTTP).
# ===========================================================================

using Dates
using Downloads
using SHA
using TOML

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

const WASHU_BASE  = "http://geoschemdata.wustl.edu/ExtData/GEOS_C180/GEOS_IT"
const DATE        = Date(2023, 6, 1)
const Nc          = 180
const DATESTR     = Dates.format(DATE, "yyyymmdd")

# ---------------------------------------------------------------------------
# Step 1: Download raw GEOS-IT C180 CTM_A1 NetCDF
# ---------------------------------------------------------------------------

function download_geosit_data(outdir::String)
    daydir = joinpath(outdir, "raw", DATESTR)
    mkpath(daydir)

    y = string(Dates.year(DATE))
    m = lpad(Dates.month(DATE), 2, '0')
    fname = "GEOSIT.$(DATESTR).CTM_A1.C$(Nc).nc"
    url = "$(WASHU_BASE)/$y/$m/$fname"
    dest = joinpath(daydir, fname)

    if isfile(dest) && filesize(dest) > 1_000_000
        @info "Raw data already cached: $dest ($(round(filesize(dest) / 1e9, digits=2)) GB)"
        return joinpath(outdir, "raw")
    end

    @info "Downloading GEOS-IT C$(Nc) CTM_A1 from WashU..."
    @info "  URL: $url"
    @info "  Destination: $dest"
    @info "  Expected size: ~4.2 GB"

    for attempt in 1:3
        try
            Downloads.download(url, dest)
            sz = round(filesize(dest) / 1e9, digits=2)
            @info "  Downloaded: $sz GB"
            return joinpath(outdir, "raw")
        catch e
            @warn "Attempt $attempt failed: $e"
            isfile(dest) && rm(dest; force=true)
            attempt < 3 && sleep(5 * attempt)
        end
    end
    error("Failed to download after 3 attempts")
end

# ---------------------------------------------------------------------------
# Step 2: Preprocess to flat binary using existing pipeline
# ---------------------------------------------------------------------------

function preprocess_data(raw_dir::String, outdir::String)
    preproc_dir = joinpath(outdir, "preprocessed")
    ft_tag = "float32"
    bin_file = joinpath(preproc_dir, "geosfp_cs_$(DATESTR)_$(ft_tag).bin")

    if isfile(bin_file) && filesize(bin_file) > 8192
        @info "Preprocessed binary already exists: $bin_file ($(round(filesize(bin_file) / 1e9, digits=2)) GB)"
        return preproc_dir
    end

    mkpath(preproc_dir)

    # Write a temporary preprocessing config
    config_path = joinpath(outdir, "preprocess_quickstart.toml")
    config = Dict(
        "product" => Dict(
            "name"         => "geosit_c180",
            "mass_flux_dt" => 450.0,
        ),
        "input" => Dict(
            "data_dir"   => raw_dir,
            "start_date" => Dates.format(DATE, "yyyy-mm-dd"),
            "end_date"   => Dates.format(DATE, "yyyy-mm-dd"),
        ),
        "output" => Dict(
            "directory" => preproc_dir,
        ),
        "grid" => Dict(
            "halo_width" => 3,
        ),
        "numerics" => Dict(
            "float_type" => "Float32",
        ),
        "diagnostics" => Dict(
            "verbose" => true,
        ),
    )
    open(config_path, "w") do io
        TOML.print(io, config)
    end

    @info "Running preprocessor..."
    @info "  Config: $config_path"
    @info "  Output: $preproc_dir"

    # Shell out to the preprocessor script (it uses @everywhere, Distributed, etc.)
    project_dir = dirname(dirname(@__FILE__))
    preproc_script = joinpath(project_dir, "scripts", "preprocess_geosfp_cs.jl")

    cmd = `$(Base.julia_cmd()) --project=$project_dir $preproc_script $config_path`
    @info "  Command: $cmd"
    run(cmd)

    isfile(bin_file) || error("Preprocessing did not produce expected output: $bin_file")
    @info "Preprocessed: $bin_file ($(round(filesize(bin_file) / 1e9, digits=2)) GB)"
    return preproc_dir
end

# ---------------------------------------------------------------------------
# Step 3: Create tarball for hosting as artifact
# ---------------------------------------------------------------------------

function create_tarball(preproc_dir::String, outdir::String)
    bin_files = filter(f -> endswith(f, ".bin"), readdir(preproc_dir; join=false))
    isempty(bin_files) && error("No .bin files found in $preproc_dir")

    tarball = joinpath(outdir, "quickstart_met_data.tar.gz")
    @info "Creating tarball: $tarball"
    @info "  Files: $(join(bin_files, ", "))"

    # Create tarball with preprocessed dir as root, renaming to "quickstart_met_data"
    # so it unpacks cleanly for the artifact system
    staging = joinpath(outdir, "quickstart_met_data")
    mkpath(staging)
    for f in bin_files
        cp(joinpath(preproc_dir, f), joinpath(staging, f); force=true)
    end
    run(`tar -czf $tarball -C $outdir quickstart_met_data`)

    # Compute SHA256
    sha = open(tarball, "r") do io
        bytes2hex(sha256(io))
    end
    sz = round(filesize(tarball) / 1e6, digits=1)
    @info "Tarball: $tarball ($sz MB)"
    @info "SHA256: $sha"

    println()
    println("=" ^ 70)
    println("Add to Artifacts.toml:")
    println("=" ^ 70)
    println("""
[quickstart_met_data]
git-tree-sha1 = "<run Pkg.Artifacts.create_artifact() to compute this>"
lazy = true

    [[quickstart_met_data.download]]
    url = "https://github.com/RemoteSensingTools/AtmosTransport/releases/download/data-v1/quickstart_met_data.tar.gz"
    sha256 = "$sha"
""")
    println("To compute git-tree-sha1:")
    println("""
    julia -e '
    using Pkg, Pkg.Artifacts
    hash = create_artifact() do dir
        run(`tar -xzf $tarball -C \$dir`)
    end
    println("git-tree-sha1 = \\"", bytes2hex(hash.bytes), "\\"")
    '
""")

    return tarball
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    outdir = length(ARGS) >= 1 ? ARGS[1] :
        joinpath(homedir(), "data", "AtmosTransport", "quickstart_build")

    println("=" ^ 70)
    println("  AtmosTransport Quickstart Artifact Builder")
    println("  GEOS-IT C$(Nc) cubed-sphere, $(Dates.format(DATE, "yyyy-mm-dd"))")
    println("=" ^ 70)
    println()

    raw_dir     = download_geosit_data(outdir)
    preproc_dir = preprocess_data(raw_dir, outdir)
    tarball     = create_tarball(preproc_dir, outdir)

    println()
    println("=" ^ 70)
    println("  Done!")
    println("  Tarball: $tarball")
    println("  Upload to GitHub Release and update Artifacts.toml")
    println("=" ^ 70)
end

main()
