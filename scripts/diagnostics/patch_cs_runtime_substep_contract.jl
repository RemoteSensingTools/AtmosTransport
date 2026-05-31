#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# patch_cs_runtime_substep_contract.jl — add the missing
# `runtime_substep_contract = "binary_schedule"` header flag to existing
# cubed-sphere transport binaries IN PLACE.
#
# Why: early ERA5 N320 → C180 binaries baked the per-window advection substep
# schedule into their data but the writer never wrote the header flag the
# runtime keys on (`uses_binary_substep_contract`). Without it the driven loop
# runs convection + chemistry once PER SUBSTEP (~25×/window) instead of once
# per met window — ~8× slower wall, convection-dominated. The writer is now
# fixed (era5_n320_regrid.jl); this patches binaries already on disk.
#
# Safe because the header is JSON in a fixed `header_bytes` region (131072 for
# these binaries — read from the binary's own `header_bytes` field, never
# assumed) terminated by a 0x00 null; the reader stops at the first null and the
# payload starts at a FIXED offset (`header_bytes`). Adding one ~50-byte key fits
# with room to spare and leaves `header_bytes` — hence the payload offset and the
# 34 GB payload itself — untouched. Only the first `header_bytes` are rewritten.
#
# Idempotent: a binary that already declares the flag is left untouched.
# NOT crash-safe: a kill mid-write leaves the header partially overwritten. Run
# on otherwise-reproducible binaries (the payload can be regenerated).
#
# Usage:
#   julia --project=. scripts/diagnostics/patch_cs_runtime_substep_contract.jl [--apply] <bin-or-glob>...
#   (default is a DRY RUN; pass --apply to write)
# ---------------------------------------------------------------------------

using JSON3

const CONTRACT_KEY = "runtime_substep_contract"
const CONTRACT_VAL = "binary_schedule"

function _read_header(path::AbstractString)
    open(path, "r") do io
        raw = read(io, min(filesize(path), 262144))
        json_end = something(findfirst(==(0x00), raw), length(raw) + 1) - 1
        hdr = JSON3.read(String(raw[1:json_end]))
        dict = Dict{String, Any}(String(k) => v for (k, v) in pairs(hdr))
        # Default matches the streaming CS writer (`header_bytes = 131072`); the
        # real value is read from the binary's own field. Guard against a header
        # region larger than the probe window so we never rewrite a truncated
        # header and corrupt the file.
        hb = Int(get(dict, "header_bytes", 131072))
        hb <= length(raw) ||
            error("header_bytes=$hb exceeds the $(length(raw))-byte probe window; " *
                  "enlarge the read in _read_header before patching $(basename(path))")
        return dict, hb
    end
end

function patch!(path::AbstractString; apply::Bool)
    dict, header_bytes = _read_header(path)
    cur = get(dict, CONTRACT_KEY, nothing)
    steps = get(dict, "steps_per_window", "?")
    if cur == CONTRACT_VAL
        @info "skip (already declared)" file = basename(path) steps_per_window = steps
        return :skip
    elseif cur !== nothing
        @warn "present but unexpected value — NOT overwriting" file = basename(path) value = cur
        return :conflict
    end

    dict[CONTRACT_KEY] = CONTRACT_VAL
    new_json = JSON3.write(dict)
    nbytes = ncodeunits(new_json)
    if nbytes + 1 > header_bytes
        @error "patched header would overflow header_bytes — skipping" file = basename(path) nbytes header_bytes
        return :overflow
    end

    if !apply
        @info "WOULD patch (dry run)" file = basename(path) steps_per_window = steps new_header_bytes = nbytes header_bytes
        return :would
    end

    # Rewrite exactly the first `header_bytes`: JSON + null terminator + zero pad.
    buf = zeros(UInt8, header_bytes)
    copyto!(buf, 1, codeunits(new_json), 1, nbytes)   # buf[nbytes+1] stays 0x00 → terminator
    open(path, "r+") do io
        seek(io, 0)
        write(io, buf)
    end

    # Verify round-trip.
    check, _ = _read_header(path)
    ok = get(check, CONTRACT_KEY, nothing) == CONTRACT_VAL
    ok || error("verification failed after patch: $path")
    @info "patched ✓" file = basename(path) steps_per_window = steps
    return :patched
end

function main(args)
    apply = "--apply" in args
    targets = filter(a -> a != "--apply", args)
    isempty(targets) && error("usage: [--apply] <bin-or-glob>...")
    paths = String[]
    for t in targets
        if isfile(t)
            push!(paths, t)
        else
            dir = dirname(t) == "" ? "." : dirname(t)
            pat = Regex("^" * replace(basename(t), "." => "\\.", "*" => ".*") * "\$")
            for f in readdir(dir)
                occursin(pat, f) && push!(paths, joinpath(dir, f))
            end
        end
    end
    isempty(paths) && error("no binaries matched: $(targets)")
    sort!(unique!(paths))
    @info "$(apply ? "APPLY" : "DRY RUN") over $(length(paths)) binaries"
    counts = Dict{Symbol, Int}()
    for p in paths
        r = patch!(p; apply = apply)
        counts[r] = get(counts, r, 0) + 1
    end
    @info "done" summary = counts
    return nothing
end

# Only run as a script; `include`-ing the file (e.g. from tests) exposes
# `patch!` / `_read_header` without executing.
if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
