#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# Convert daily binary output files to compressed NetCDF4
#
# Handles both single files and directories of daily files.
# Supports native cubed-sphere and lat-lon grids automatically
# (detected from the JSON header in each .bin file).
#
# Usage:
#   # Convert all daily files in a directory:
#   julia --project=. scripts/convert_output_to_netcdf.jl /temp2/catrine-runs/output/
#
#   # Convert a single file:
#   julia --project=. scripts/convert_output_to_netcdf.jl output_20211201.bin
#
#   # With compression:
#   julia --project=. scripts/convert_output_to_netcdf.jl --deflate=4 /temp2/catrine-runs/output/
# ---------------------------------------------------------------------------

using AtmosTransport
using AtmosTransport.IO: convert_binary_to_netcdf

function main()
    if isempty(ARGS)
        println(stderr, """
Usage: julia --project=. scripts/convert_output_to_netcdf.jl [--deflate=N] <path>

  path      Directory of .bin files or a single .bin file
  --deflate NetCDF deflate level (0-9, default: 0)
""")
        exit(1)
    end

    # Parse args
    deflate_level = 0
    paths = String[]
    for arg in ARGS
        if startswith(arg, "--deflate=")
            deflate_level = parse(Int, split(arg, "=")[2])
        else
            push!(paths, arg)
        end
    end

    isempty(paths) && error("No path specified")
    target = paths[1]

    # Collect .bin files
    bin_files = if isdir(target)
        sort(filter(f -> endswith(f, ".bin"), readdir(target; join=true)))
    elseif isfile(target) && endswith(target, ".bin")
        [target]
    else
        error("Not a .bin file or directory: $target")
    end

    isempty(bin_files) && error("No .bin files found in $target")

    @info "Converting $(length(bin_files)) binary file(s) to NetCDF (deflate=$deflate_level)"

    n_done = 0
    n_skip = 0
    n_fail = 0
    t0 = time()

    # NCDatasets/HDF5-C is not thread-safe, so files are converted sequentially.
    for (idx, bin_path) in enumerate(bin_files)
        nc_path = splitext(bin_path)[1] * ".nc"
        if isfile(nc_path)
            @info "  [$idx/$(length(bin_files))] Skipping $(basename(bin_path)) — NC exists"
            n_skip += 1
            continue
        end
        try
            convert_binary_to_netcdf(bin_path; nc_path, deflate_level)
            n_done += 1
        catch e
            @error "  [$idx/$(length(bin_files))] Failed: $(basename(bin_path))" exception=e
            n_fail += 1
        end
    end

    elapsed = time() - t0
    @info """
    Conversion complete.
    ====================
    Converted: $n_done / $(length(bin_files))
    Skipped:   $n_skip
    Failed:    $n_fail
    Elapsed:   $(round(elapsed; digits=1))s
    """
end

main()
