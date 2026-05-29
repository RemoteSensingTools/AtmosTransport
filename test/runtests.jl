#!/usr/bin/env julia
#
# Main test suite entrypoint for AtmosTransport.
#
# Tiered layout (see test/README.md):
#   test/core/        — canonical CI baseline (no external data); must stay green
#   test/real_data/   — needs preprocessed binaries / ERA5 GRIB; opt-in via --all
#   test/diagnostic/  — large numerical sweeps; opt-in via --diagnostic or --all
#   test/archived/    — never run; kept as reference (see archived/legacy_README.md)
#   test/orphan/      — promotion candidates; CI-excluded; opt-in via --orphan
#   test/regridding/  — semantic grouping with its own (optional) runtests.jl
#
# Usage:
#   julia --project=. test/runtests.jl                # core only (CI default)
#   julia --project=. test/runtests.jl --all          # core + real_data + diagnostic
#   julia --project=. test/runtests.jl --diagnostic   # core + diagnostic
#   julia --project=. test/runtests.jl --orphan       # core + orphan watchlist
#   julia --project=. test/runtests.jl --tiers=core,real_data,orphan
#
# Each test file is included into a fresh anonymous module so files don't
# leak `using .AtmosTransport.X: Y` bindings into each other. On Julia 1.12
# top-level `include`s in `Main` silently keep stale bindings between files
# (e.g. `op isa AbstractConvection` returning false because LHS is from
# include #2 but the RHS abstract type came from include #1).

const TIER_FOLDERS = (
    core       = "core",
    real_data  = "real_data",
    diagnostic = "diagnostic",
    orphan     = "orphan",
)

function _selected_tiers(args::Vector{String})
    selected = Set{Symbol}((:core,))
    for arg in args
        if arg == "--all"
            push!(selected, :real_data); push!(selected, :diagnostic)
        elseif arg == "--diagnostic"
            push!(selected, :diagnostic)
        elseif arg == "--real-data" || arg == "--real_data"
            push!(selected, :real_data)
        elseif arg == "--orphan"
            push!(selected, :orphan)
        elseif startswith(arg, "--tiers=")
            for s in split(arg[length("--tiers=")+1:end], ",")
                key = Symbol(replace(strip(s), "-" => "_"))
                haskey(TIER_FOLDERS, key) ||
                    error("Unknown tier '$key' (known: $(keys(TIER_FOLDERS)))")
                push!(selected, key)
            end
        end
    end
    return selected
end

function _tier_files(tier::Symbol)
    folder = joinpath(@__DIR__, TIER_FOLDERS[tier])
    isdir(folder) || return String[]
    return sort([joinpath(TIER_FOLDERS[tier], f)
                 for f in readdir(folder) if endswith(f, ".jl")])
end

function run_test_file_isolated(test_file::AbstractString)
    mod_name = Symbol("Test_", replace(replace(basename(test_file), "." => "_"), "/" => "_"))
    mod = Module(mod_name)
    Core.eval(mod, :(include(path::AbstractString) = Base.include($mod, path)))
    Core.eval(mod, :(include(mapexpr::Function, path::AbstractString) = Base.include(mapexpr, $mod, path)))
    Core.eval(mod, :(eval(expr) = Core.eval($mod, expr)))
    return Base.include(mod, joinpath(@__DIR__, test_file))
end

selected = _selected_tiers(ARGS)
for tier in (:core, :real_data, :diagnostic, :orphan)
    tier in selected || continue
    files = _tier_files(tier)
    if isempty(files)
        @info "Tier $(tier): no files"
        continue
    end
    @info "── Tier $(tier) — $(length(files)) files ──"
    for f in files
        @info "Running $f"
        run_test_file_isolated(f)
    end
end

skipped = setdiff(Set(keys(TIER_FOLDERS)), selected)
isempty(skipped) || @info "Skipped tiers: $(sort(collect(skipped))) (opt-in flags: --real-data --diagnostic --orphan --all)"
@info "Test suite complete."
