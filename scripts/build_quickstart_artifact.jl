#!/usr/bin/env julia
# ===========================================================================
# Build quickstart data: download GEOS-IT C180 from WashU → preprocess
# to flat binary → preprocess EDGAR CO2 → create tarball for GitHub Release.
#
# Usage (maintainer only):
#   julia --project=. scripts/build_quickstart_artifact.jl [output_dir]
#
# Steps:
#   1. Download 1 day of GEOS-IT C180 CTM_A1 from WashU archive (~4.2 GB)
#   2. Preprocess to flat binary via preprocess_geosfp_cs.jl (~4 GB)
#   3. Truncate to 12 hours (keeps tarball < 2 GB GitHub Release limit)
#   4. Preprocess EDGAR v8.0 CO2 emissions to C180 cubed-sphere binary
#   5. Package everything as tarball (~1.4 GB compressed)
#
# No authentication required (WashU archive is public HTTP).
# ===========================================================================

using Dates
using Downloads
using JSON3
using SHA
using TOML

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

const WASHU_BASE   = "http://geoschemdata.wustl.edu/ExtData/GEOS_C180/GEOS_IT"
const DATE         = Date(2023, 6, 1)
const Nc           = 180
const DATESTR      = Dates.format(DATE, "yyyymmdd")
const N_WINDOWS    = 12        # 12 hours (of 24 in the daily file) — keeps tarball < 2 GB
const HEADER_SIZE  = 8192

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
        return (joinpath(outdir, "raw"), dest)
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
            return (joinpath(outdir, "raw"), dest)
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
# Step 2b: Truncate binary to N_WINDOWS (keeps tarball under 2 GB)
# ---------------------------------------------------------------------------

function truncate_binary!(bin_path::String, n_windows::Int)
    # Read header JSON
    header_bytes = Vector{UInt8}(undef, HEADER_SIZE)
    open(bin_path, "r") do io
        read!(io, header_bytes)
    end
    json_end = something(findfirst(==(0x00), header_bytes), HEADER_SIZE + 1) - 1
    header = JSON3.read(String(header_bytes[1:json_end]))

    old_nt = Int(header[:Nt])
    if n_windows >= old_nt
        @info "Binary already has $old_nt windows — no truncation needed"
        return
    end

    window_bytes = Int(header[:window_bytes])
    new_size = HEADER_SIZE + n_windows * window_bytes

    # Rewrite header with updated Nt
    header_dict = Dict{String,Any}(String(k) => v for (k, v) in pairs(header))
    header_dict["Nt"] = n_windows
    new_json = JSON3.write(header_dict)

    new_header = zeros(UInt8, HEADER_SIZE)
    copyto!(new_header, 1, Vector{UInt8}(new_json), 1, length(new_json))

    open(bin_path, "r+") do io
        write(io, new_header)
        truncate(io, new_size)
    end

    @info "Truncated: $old_nt → $n_windows windows ($(round(new_size / 1e9, digits=2)) GB)"
end

# ---------------------------------------------------------------------------
# Step 3: Preprocess EDGAR CO2 to C180 cubed-sphere binary
# ---------------------------------------------------------------------------

function preprocess_edgar(nc_ref_file::String, outdir::String)
    edgar_bin = joinpath(outdir, "preprocessed", "edgar_co2_cs_c180_float32.bin")

    if isfile(edgar_bin) && filesize(edgar_bin) > 4096
        @info "EDGAR binary already exists: $edgar_bin ($(round(filesize(edgar_bin) / 1e6, digits=1)) MB)"
        return edgar_bin
    end

    @info "Preprocessing EDGAR CO2 → C180 cubed-sphere..."

    project_dir = dirname(dirname(@__FILE__))
    edgar_script = joinpath(project_dir, "scripts", "preprocess_edgar_cs.jl")

    # Find EDGAR source NetCDF
    edgar_file = expanduser("~/data/emissions/edgar_v8/v8.0_FT2022_GHG_CO2_2022_TOTALS_emi.nc")
    isfile(edgar_file) || error("""
        EDGAR source NetCDF not found: $edgar_file
        Download from: https://edgar.jrc.ec.europa.eu/dataset_ghg80
        Expected: v8.0_FT2022_GHG_CO2_2022_TOTALS_emi.nc (~26 MB)
    """)

    env = copy(ENV)
    env["EDGAR_FILE"]      = edgar_file
    env["OUTFILE"]         = edgar_bin
    env["NC_GRID"]         = string(Nc)
    env["GEOSFP_REF_FILE"] = nc_ref_file

    cmd = setenv(`$(Base.julia_cmd()) --project=$project_dir $edgar_script`, env)
    @info "  Command: $cmd"
    run(cmd)

    isfile(edgar_bin) || error("EDGAR preprocessing did not produce: $edgar_bin")
    @info "EDGAR binary: $edgar_bin ($(round(filesize(edgar_bin) / 1e6, digits=1)) MB)"
    return edgar_bin
end

# ---------------------------------------------------------------------------
# Step 4: Create tarball for hosting on GitHub Releases
# ---------------------------------------------------------------------------

function create_tarball(preproc_dir::String, outdir::String)
    bin_files = filter(f -> endswith(f, ".bin"), readdir(preproc_dir; join=false))
    isempty(bin_files) && error("No .bin files found in $preproc_dir")

    tarball = joinpath(outdir, "quickstart_met_data.tar.gz")
    @info "Creating tarball: $tarball"
    @info "  Files: $(join(bin_files, ", "))"

    # Stage files into a clean directory matching the tarball name
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

    return tarball
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    outdir = length(ARGS) >= 1 ? ARGS[1] :
        joinpath(homedir(), "data", "AtmosTransport", "quickstart_build")

    println("=" ^ 70)
    println("  AtmosTransport Quickstart Data Builder")
    println("  GEOS-IT C$(Nc) cubed-sphere, $(Dates.format(DATE, "yyyy-mm-dd"))")
    println("  Includes: met data (12h) + EDGAR CO2 emissions")
    println("=" ^ 70)
    println()

    raw_dir, nc_ref_file = download_geosit_data(outdir)
    preproc_dir = preprocess_data(raw_dir, outdir)

    # Truncate to N_WINDOWS to keep tarball under GitHub's 2 GB limit
    ft_tag = "float32"
    bin_file = joinpath(preproc_dir, "geosfp_cs_$(DATESTR)_$(ft_tag).bin")
    truncate_binary!(bin_file, N_WINDOWS)

    # Preprocess EDGAR CO2 emissions for C180
    preprocess_edgar(nc_ref_file, outdir)

    tarball = create_tarball(preproc_dir, outdir)

    println()
    println("=" ^ 70)
    println("  Done!")
    println("  Tarball: $tarball")
    println()
    println("  To publish:")
    println("    1. Upload $tarball to GitHub Releases:")
    println("       https://github.com/RemoteSensingTools/AtmosTransport/releases/tag/data-v1")
    println("    2. Users download and extract:")
    println("       mkdir -p ~/data/AtmosTransport")
    println("       tar -xzf quickstart_met_data.tar.gz -C ~/data/AtmosTransport/")
    println("=" ^ 70)
end

main()
