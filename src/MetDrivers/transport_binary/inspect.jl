# Read-only inspection: binary_capabilities, inspect_binary, header peeking, capability print.
#
# Part of the TransportBinary format implementation; included from
# `TransportBinary.jl` into the `MetDrivers` module (shared namespace,
# shared `using`s). Split out of the former 2658-line monolith — pure code
# move, no behavior change.

"""
    binary_capabilities(reader) -> NamedTuple

Summarise what operators this binary can drive. Works on either
`TransportBinaryReader` (LL/RG) or `CubedSphereBinaryReader` (CS) via
the existing `has_flux_delta`, `has_tm5_convection`, `has_cmfmc`, and
`has_qv` predicates. Fields:

- `advection :: Bool` — always `true` (m, am, bm, cm are required).
- `replay_gate :: Bool` — plan-39 dam/dbm/dcm/dm present.
- `tm5_convection :: Bool` — entu/detu/entd/detd all present.
- `cmfmc_convection :: Bool` — cmfmc present (CS only; LL/RG returns false).
- `surface_pressure :: Bool` — ps present.
- `humidity :: Bool` — qv or qv_start/qv_end present.
- `mass_basis :: Symbol` — `:dry` or `:moist`.
- `grid_type :: Symbol` — `:latlon` / `:reduced_gaussian` / `:cubed_sphere`.
- `payload_sections :: Vector{Symbol}` — raw set for debugging.
"""
function binary_capabilities(reader::TransportBinaryReader)
    hdr = reader.header
    return (
        advection        = all(s in hdr.payload_sections for s in (:m, :am, :bm, :cm)),
        replay_gate      = has_flux_delta(reader),
        tm5_convection   = has_tm5_convection(reader),
        cmfmc_convection = has_cmfmc(reader),
        pbl_diffusion    = has_surface(reader),
        gchp_vdiff       = has_surface(reader) && has_vdiff_fields(reader),
        surface_pressure = :ps in hdr.payload_sections,
        humidity         = has_qv(reader),
        mass_basis       = hdr.mass_basis,
        grid_type        = Symbol(hdr.grid_type),
        nlevel           = hdr.nlevel,
        steps_per_window = hdr.steps_per_window,
        variable_step_schedule = _has_variable_step_schedule(hdr.steps_per_window_by_window),
        preprocessor_contract = nothing,
        vertical_Nz_output = nothing,
        adaptive_substeps = nothing,
        payload_sections = hdr.payload_sections,
    )
end

"""
    inspect_binary(path; io = stdout) -> NamedTuple

Open the binary at `path` (auto-detecting LL/RG vs CS format), print
a capability-augmented report to `io`, and return the
`binary_capabilities` NamedTuple for programmatic consumption (tests,
CLI capability-intersection, folder-level validation).

Obsolete `format_version < TRANSPORT_BINARY_FORMAT_VERSION` files are rejected
here just like runtime drivers; inspect the raw JSON header with external tools
if a stale file must be audited.
"""
function inspect_binary(path::AbstractString; io::IO = stdout)
    isfile(path) || throw(ArgumentError("binary not found: $(path)"))
    reader = _open_binary_for_inspection(path)
    try
        println(io, reader)
        println(io)
        _print_capability_rows(io, reader)
        return binary_capabilities(reader)
    finally
        close(reader)
    end
end

# Internal: open either LL/RG or CS reader. Peek at the JSON header
# `grid_type` field to pick; a CS binary opened as `TransportBinaryReader`
# errors during semantics validation, which is an unhelpful failure
# mode for a diagnostic tool.
function _open_binary_for_inspection(path::AbstractString)
    grid_type_hint = _peek_grid_type(path)
    if grid_type_hint === :cubed_sphere
        return _open_cubed_sphere_binary_reader(path)
    end
    return TransportBinaryReader(path; FT = Float64)
end

# Peek the JSON header to decide LL/RG vs CS without constructing a
# full reader. Both LL/RG and CS writers start the file with a
# null-terminated JSON blob at byte 0 (see
# `_parse_transport_header` and `CubedSphereBinaryReader` — both find
# the first `0x00` and parse the preceding bytes). We only need
# `grid_type`; on any parse failure we fall back to `:latlon` so the
# `TransportBinaryReader` constructor can emit its own richer error.
function _peek_grid_type(path::AbstractString)
    try
        open(path, "r") do io
            chunk = read(io, min(filesize(path), 262144))
            null_idx = findfirst(==(0x00), chunk)
            json_end = null_idx === nothing ? length(chunk) : null_idx - 1
            json_end < 1 && return :latlon
            hdr = _peek_parse_header(@view chunk[1:json_end])
            raw = get(hdr, :grid_type, get(hdr, "grid_type", "latlon"))
            return Symbol(lowercase(String(raw)))
        end
    catch
        return :latlon
    end
end

# Lightweight JSON parse that doesn't pull in the reader's full stack.
# Falls back gracefully if the header format differs (e.g. legacy
# binaries that don't use JSON).
function _peek_parse_header(bytes)
    try
        return JSON3.read(String(bytes))
    catch
        return Dict{String, Any}()
    end
end

# Forward declaration; the CS reader lives in CubedSphereBinaryReader.jl
# and is not a subtype of anything common. The MetDrivers module's
# load order (TransportBinary.jl first, CubedSphereBinaryReader.jl
# second) means we must stub this here and let the CS file's include
# define the method.
function _open_cubed_sphere_binary_reader end

function _print_capability_rows(io::IO, reader)
    caps = binary_capabilities(reader)
    println(io, "Capabilities:")
    _print_cap(io, caps.advection,        "advection",        "(m, am, bm, cm)")
    _print_cap(io, caps.replay_gate,      "plan-39 replay",   "(dam, dbm, dcm, dm)")
    _print_cap(io, caps.tm5_convection,   "TM5 convection",   "(entu, detu, entd, detd)")
    _print_cap(io, caps.cmfmc_convection, "CMFMC convection", "(cmfmc)")
    _print_cap(io, caps.pbl_diffusion,    "PBL diffusion",    "(pblh, ustar, pbl_hflux, t2m)")
    _print_cap(io, caps.surface_pressure, "surface pressure", "(ps)")
    _print_cap(io, caps.humidity,         "humidity",         "(qv or qv_start/qv_end)")
    println(io, "  mass_basis       = ", caps.mass_basis)
    println(io, "  grid_type        = ", caps.grid_type)
end

@inline function _print_cap(io::IO, present::Bool, label::AbstractString, ingredients::AbstractString)
    mark = present ? "✓" : "✗"
    @printf(io, "  %s %-16s %s\n", mark, label, ingredients)
end
