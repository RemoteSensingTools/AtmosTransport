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

# Refuse to declare the contract on anything that isn't the intended target: a
# cubed-sphere transport binary that actually carries the per-window substep
# schedule the runtime keys on. `binary_schedule` is the panel-native CS substep
# contract, and the runtime's schedule parser REQUIRES `steps_per_window_by_window`
# — declaring the flag on a LL/RG binary, or on a CS binary that lacks the schedule
# array, would make the driven loop throw at load (strictly worse than the
# slow-but-correct unpatched state). Validate before touching the header.
function _validate_patch_target(dict, path)
    gt = get(dict, "grid_type", nothing)
    String(something(gt, "")) == "cubed_sphere" || error(
        "$(basename(path)): grid_type=$(repr(gt)) — this patcher only applies to " *
        "cubed_sphere transport binaries (binary_schedule is the panel-native CS " *
        "substep contract). Refusing to patch.")
    haskey(dict, "steps_per_window_by_window") || error(
        "$(basename(path)): header has no `steps_per_window_by_window` — the runtime's " *
        "binary_schedule contract requires the per-window schedule array, so declaring " *
        "it here would make the driven loop throw at load. This binary predates the " *
        "schedule writer; regenerate it instead of patching.")
    # Mirror the exact invariants `_parse_steps_per_window_schedule` enforces at
    # load (header.jl), so a successful patch GUARANTEES the runtime will accept the
    # schedule — never the "patched but throws at load" state for a malformed one.
    sched = collect(dict["steps_per_window_by_window"])
    all(x -> x isa Real && x > 0 && x == floor(x), sched) || error(
        "$(basename(path)): `steps_per_window_by_window` must be all positive integers; " *
        "got $(sched). Refusing to declare a contract over a broken schedule.")
    nwindow = get(dict, "nwindow", nothing)
    nwindow === nothing && error(
        "$(basename(path)): header lacks `nwindow`; cannot verify the schedule length.")
    length(sched) == Int(nwindow) || error(
        "$(basename(path)): schedule length $(length(sched)) ≠ nwindow $(nwindow) — the " *
        "runtime rejects this. Regenerate the binary instead of patching.")
    spw = get(dict, "steps_per_window", nothing)
    spw === nothing && error(
        "$(basename(path)): header lacks `steps_per_window`; cannot verify the schedule maximum.")
    Int(spw) == Int(maximum(sched)) || error(
        "$(basename(path)): steps_per_window=$(spw) ≠ maximum(schedule)=$(maximum(sched)) — " *
        "the runtime requires equality. Regenerate the binary instead of patching.")
    return nothing
end

function patch!(path::AbstractString; apply::Bool)
    dict, header_bytes = _read_header(path)
    _validate_patch_target(dict, path)
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
