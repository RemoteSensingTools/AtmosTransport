#!/usr/bin/env julia
# ===========================================================================
# Compare our reference output with TM5 output (Option A validation).
#
# Loads two NetCDF files (ours and TM5), aligns dimensions if needed,
# computes RMSE, correlation, max diff. Delegates menial work to this script.
#
# Usage:
#   julia --project=. scripts/compare_tm5_output.jl our_output.nc tm5_output.nc
#
# Optional env:
#   OUR_VAR   = variable name in our file (default: tracer_c or c)
#   TM5_VAR   = variable name in TM5 file (default: same or CO2 or tracer)
#   TIME_IDX  = time index to compare (default: last)
# ===========================================================================

using NCDatasets
using LinearAlgebra: norm, dot

function main()
    args = ARGS
    if length(args) < 2
        println("Usage: julia compare_tm5_output.jl our_output.nc tm5_output.nc")
        println("  Optional env: OUR_VAR, TM5_VAR, TIME_IDX")
        exit(1)
    end
    our_nc = args[1]
    tm5_nc = args[2]
    our_var = get(ENV, "OUR_VAR", "tracer_c")
    tm5_var = get(ENV, "TM5_VAR", "tracer_c")  # adjust if TM5 uses different name
    time_idx = parse(Int, get(ENV, "TIME_IDX", "0"))  # 0 = last

    for f in [our_nc, tm5_nc]
        isfile(f) || error("File not found: $f")
    end

    ds_our = NCDataset(our_nc)
    ds_tm5 = NCDataset(tm5_nc)

    # Resolve variable names
    our_name = haskey(ds_our, our_var) ? our_var : (haskey(ds_our, "c") ? "c" : first(keys(ds_our)))
    if !haskey(ds_tm5, tm5_var)
        cand = [k for k in keys(ds_tm5) if ndims(ds_tm5[k]) >= 3]
        tm5_name = isempty(cand) ? error("No 3D variable in TM5 file") : cand[1]
    else
        tm5_name = tm5_var
    end

    A = ds_our[our_name][:]
    B = ds_tm5[tm5_name][:]

    # Time index: if 4D (lon,lat,lev,time), take last or TIME_IDX
    if ndims(A) == 4
        tidx = time_idx <= 0 ? size(A, 4) : time_idx
        A = A[:, :, :, tidx]
    end
    if ndims(B) == 4
        tidx = time_idx <= 0 ? size(B, 4) : time_idx
        B = B[:, :, :, tidx]
    end

    A = Float64.(A)
    B = Float64.(B)

    if size(A) != size(B)
        println("Shape mismatch: our $(size(A)) vs TM5 $(size(B)). Regridding not implemented; use same grid or regrid externally.")
        exit(1)
    end

    A = A[:]
    B = B[:]
    n = length(A)
    rmse = sqrt(sum((A .- B) .^ 2) / n)
    mae = sum(abs.(A .- B)) / n
    mx = maximum(abs.(A .- B))
    cA = A .- sum(A) / n
    cB = B .- sum(B) / n
    corr = dot(cA, cB) / (norm(cA) * norm(cB) + 1e-20)

    println("Comparison (our=$our_nc vs TM5=$tm5_nc)")
    println("  RMSE:      $rmse")
    println("  MAE:       $mae")
    println("  Max diff:  $mx")
    println("  Correlation: $corr")

    close(ds_our)
    close(ds_tm5)
end

main()
