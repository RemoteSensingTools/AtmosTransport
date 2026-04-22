#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# ERA5 physics NC → mmap-friendly BIN converter (plan 24 Commit 2 CLI).
#
# Builds one calendar-day BIN per call. Given a date range, iterates
# days and converts each. Idempotent: skips days whose BIN already
# exists unless --force is given.
#
# Usage:
#   julia --project=. scripts/preprocessing/convert_era5_physics_nc_to_bin.jl \
#       --nc-dir  ~/data/AtmosTransport/met/era5/0.5x0.5/physics \
#       --bin-dir /temp1/met/era5/0.5x0.5/physics_bin \
#       --start   2021-12-02 \
#       --end     2021-12-05 \
#       [--force] [--quiet]
#
# Note: building a BIN for date D requires the convection NC for
# BOTH D-1 and D (calendar-day splicing). For a date range
# start..end, make sure the archive also has `start - 1 day`'s
# convection NC.
# ---------------------------------------------------------------------------

using Dates

# Load package last to let the arg-parser error out quickly on typos.
include(joinpath(@__DIR__, "..", "..", "src", "AtmosTransport.jl"))
using .AtmosTransport.Preprocessing: convert_era5_physics_nc_to_bin

function parse_args(argv::Vector{String})
    nc_dir  = nothing
    bin_dir = nothing
    start_s = nothing
    end_s   = nothing
    force   = false
    quiet   = false
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--nc-dir"
            nc_dir = argv[i + 1]; i += 2
        elseif a == "--bin-dir"
            bin_dir = argv[i + 1]; i += 2
        elseif a == "--start"
            start_s = argv[i + 1]; i += 2
        elseif a == "--end"
            end_s = argv[i + 1]; i += 2
        elseif a == "--force"
            force = true; i += 1
        elseif a == "--quiet"
            quiet = true; i += 1
        elseif a == "-h" || a == "--help"
            print_help_and_exit()
        else
            @error "Unknown argument: $a"
            print_help_and_exit(; status = 2)
        end
    end
    (nc_dir  === nothing) && (missing_arg("--nc-dir"))
    (bin_dir === nothing) && missing_arg("--bin-dir")
    (start_s === nothing) && missing_arg("--start")
    (end_s   === nothing) && missing_arg("--end")
    return (nc_dir  = nc_dir,
            bin_dir = bin_dir,
            start_date = Date(start_s),
            end_date   = Date(end_s),
            force   = force,
            quiet   = quiet)
end

function missing_arg(name::String)
    @error "Missing required argument: $name"
    print_help_and_exit(; status = 2)
end

function print_help_and_exit(; status::Int = 0)
    println(stderr, """
Usage: convert_era5_physics_nc_to_bin.jl --nc-dir DIR --bin-dir DIR \\
                                          --start YYYY-MM-DD --end YYYY-MM-DD \\
                                          [--force] [--quiet]

Build calendar-day ERA5 physics BIN files in `bin-dir/YYYY/` from
per-day convection + thermo NCs in `nc-dir`. Skip existing BINs
unless --force. Requires convection NC for D-1 as well as D.

Options:
  --nc-dir DIR    directory with era5_convection_*.nc + era5_thermo_ml_*.nc
  --bin-dir DIR   output staging root (YYYY subdir is created automatically)
  --start DATE    first calendar day (YYYY-MM-DD)
  --end   DATE    last calendar day (YYYY-MM-DD, inclusive)
  --force         overwrite existing BIN files
  --quiet         suppress per-day progress logs
""")
    exit(status)
end

function main(argv::Vector{String})
    args = parse_args(argv)
    d    = args.start_date
    done = 0
    failed  = String[]
    while d <= args.end_date
        try
            convert_era5_physics_nc_to_bin(
                args.nc_dir, args.bin_dir, d;
                force_rewrite = args.force,
                verbose = !args.quiet)
            done += 1
        catch e
            push!(failed, string(d, ": ", sprint(showerror, e)))
        end
        d += Day(1)
    end
    println(stderr,
            "\nConverted $done day(s); $(length(failed)) failure(s).")
    for msg in failed
        println(stderr, "  FAIL: ", msg)
    end
    exit(isempty(failed) ? 0 : 1)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(copy(ARGS))
end
