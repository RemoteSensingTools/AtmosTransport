#!/usr/bin/env julia
# ===========================================================================
# Download GCP-GridFED v2024.0 fossil CO₂ emissions for CATRINE
#
# Downloads yearly ZIP archives from the UEA OPeNDAP server and extracts
# the NetCDF files. Each year is ~1.3 GB compressed.
#
# Source: Jones et al. (2024), GCP-GridFED v2024.0
#         Zenodo: https://zenodo.org/records/13909046
#         OPeNDAP: http://opendap.uea.ac.uk/opendap/hyrax/greenocean/GridFED/
#         CATRINE protocol refers to v2024.1: https://zenodo.org/records/8386803
#
# Usage:
#   julia scripts/download_gridfed_catrine.jl [dest_dir] [years...]
#
# Examples:
#   julia scripts/download_gridfed_catrine.jl                    # defaults: 2021-2023
#   julia scripts/download_gridfed_catrine.jl ~/data/gridfed     # custom dir
#   julia scripts/download_gridfed_catrine.jl ~/data/gridfed 2022 2023  # specific years
# ===========================================================================

using Downloads

const BASE_URL = "http://opendap.uea.ac.uk/opendap/hyrax/greenocean/GridFED/GridFEDv2024.0"
const DEFAULT_DEST = joinpath(homedir(), "data", "AtmosTransport", "catrine", "gridfed")
const DEFAULT_YEARS = [2021, 2022, 2023]   # CATRINE period: Dec 2021 – Dec 2023

function download_gridfed(dest_dir::String, years::Vector{Int})
    mkpath(dest_dir)

    println("=" ^ 70)
    println("  GCP-GridFED v2024.0 — Fossil CO₂ Download")
    println("  Years: $(join(years, ", "))")
    println("  Destination: $dest_dir")
    println("=" ^ 70)
    println()

    for year in years
        zipname = "GCP-GridFEDv2024.0_$(year).zip"
        url = "$BASE_URL/$zipname"
        zip_dest = joinpath(dest_dir, zipname)

        # Check if already extracted (any .nc file for this year)
        existing_ncs = filter(readdir(dest_dir)) do f
            endswith(f, ".nc") && contains(f, string(year))
        end

        if !isempty(existing_ncs)
            @info "Year $year: already extracted ($(length(existing_ncs)) NC files)"
            continue
        end

        # Download ZIP
        if isfile(zip_dest) && filesize(zip_dest) > 1_000_000
            @info "Year $year: ZIP already cached ($(round(filesize(zip_dest) / 1e9, digits=2)) GB)"
        else
            @info "Year $year: downloading $zipname (~1.3 GB)..."
            for attempt in 1:3
                try
                    Downloads.download(url, zip_dest; timeout=600.0)
                    sz = round(filesize(zip_dest) / 1e9, digits=2)
                    @info "  Downloaded: $sz GB"
                    break
                catch e
                    @warn "  Attempt $attempt failed: $e"
                    isfile(zip_dest) && filesize(zip_dest) < 1_000_000 && rm(zip_dest; force=true)
                    if attempt == 3
                        error("Failed to download $zipname after 3 attempts. " *
                              "Try downloading manually from: $url")
                    end
                    sleep(10 * attempt)
                end
            end
        end

        # Extract ZIP
        @info "Year $year: extracting..."
        run(`unzip -o -d $dest_dir $zip_dest`)

        # Verify extraction
        new_ncs = filter(readdir(dest_dir)) do f
            endswith(f, ".nc") && contains(f, string(year))
        end
        @info "Year $year: extracted $(length(new_ncs)) NetCDF file(s)"
    end

    # Summary
    println()
    println("=" ^ 70)
    println("  Download complete!")
    println()
    all_ncs = sort(filter(f -> endswith(f, ".nc"), readdir(dest_dir)))
    for f in all_ncs
        sz = round(filesize(joinpath(dest_dir, f)) / 1e9, digits=2)
        println("  $f  ($sz GB)")
    end
    println()
    println("  Total: $(length(all_ncs)) files")
    example = isempty(all_ncs) ? "*.nc" : all_ncs[1]
    println("  To inspect: ncdump -h $(joinpath(dest_dir, example))")
    println("=" ^ 70)
end

function main()
    dest_dir = length(ARGS) >= 1 ? expanduser(ARGS[1]) : DEFAULT_DEST
    years = if length(ARGS) >= 2
        [parse(Int, a) for a in ARGS[2:end]]
    else
        DEFAULT_YEARS
    end
    download_gridfed(dest_dir, years)
end

main()
