# Transport-binary write/read contracts and their validators
# (TransportBinaryContract, canonical_window_constant_contract, validate_cs_writer_contract!, validate_transport_contract!).
#
# Part of the TransportBinary format implementation; included from
# `TransportBinary.jl` into the `MetDrivers` module (shared namespace,
# shared `using`s). Split out of the former 2658-line monolith — pure code
# move, no behavior change.

"""
    TransportBinaryContract(; source_flux_sampling, air_mass_sampling,
                              flux_sampling, flux_kind, delta_semantics,
                              humidity_sampling,
                              poisson_balance_target_scale,
                              poisson_balance_target_semantics)

Self-describing transport-binary timing/basis contract. All eight fields
are required — no defaults — so a writer cannot produce an ambiguous
binary. Readers call [`validate_transport_contract!`](@ref) on the parsed
header to decide whether the file is trustworthy.

Canonical usage: construct via
[`canonical_window_constant_contract`](@ref) for the memo-37 path
(`tracer drift = 0` on uniform IC for Upwind over 2 days).

Symbol fields are validated against the `_TRANSPORT_ALLOWED_*` tuples at
construction time. Combinations are also checked:
- `delta_semantics === :forward_window_endpoint_difference` requires the
  payload to carry `dm` (or `dm + dhflux`); the writer is responsible for
  honoring this.
- `humidity_sampling === :window_endpoints` requires `qv_start` + `qv_end`
  in the payload; `:single_field` requires `qv`; `:none` requires neither.
"""
struct TransportBinaryContract
    source_flux_sampling             :: Symbol
    air_mass_sampling                :: Symbol
    flux_sampling                    :: Symbol
    flux_kind                        :: Symbol
    delta_semantics                  :: Symbol
    humidity_sampling                :: Symbol
    poisson_balance_target_scale     :: Float64
    poisson_balance_target_semantics :: String

    function TransportBinaryContract(source_flux_sampling::Symbol,
                                     air_mass_sampling::Symbol,
                                     flux_sampling::Symbol,
                                     flux_kind::Symbol,
                                     delta_semantics::Symbol,
                                     humidity_sampling::Symbol,
                                     poisson_balance_target_scale::Real,
                                     poisson_balance_target_semantics::AbstractString)
        sfs = _transport_validate_source_flux_sampling(source_flux_sampling)
        ams = _transport_normalize_symbol(air_mass_sampling)
        fs  = _transport_normalize_symbol(flux_sampling)
        fk  = _transport_normalize_symbol(flux_kind)
        ds  = _transport_normalize_symbol(delta_semantics)
        hs  = _transport_normalize_symbol(humidity_sampling)
        ams in _TRANSPORT_ALLOWED_AIR_MASS_SAMPLINGS ||
            throw(ArgumentError("air_mass_sampling=$(ams) not in $(Tuple(_TRANSPORT_ALLOWED_AIR_MASS_SAMPLINGS))"))
        fs in _TRANSPORT_ALLOWED_FLUX_SAMPLINGS ||
            throw(ArgumentError("flux_sampling=$(fs) not in $(Tuple(_TRANSPORT_ALLOWED_FLUX_SAMPLINGS))"))
        fk in _TRANSPORT_ALLOWED_FLUX_KINDS ||
            throw(ArgumentError("flux_kind=$(fk) not in $(Tuple(_TRANSPORT_ALLOWED_FLUX_KINDS))"))
        ds in _TRANSPORT_ALLOWED_DELTA_SEMANTICS ||
            throw(ArgumentError("delta_semantics=$(ds) not in $(Tuple(_TRANSPORT_ALLOWED_DELTA_SEMANTICS))"))
        hs in _TRANSPORT_ALLOWED_HUMIDITY_SAMPLINGS ||
            throw(ArgumentError("humidity_sampling=$(hs) not in $(Tuple(_TRANSPORT_ALLOWED_HUMIDITY_SAMPLINGS))"))
        Float64(poisson_balance_target_scale) > 0 ||
            throw(ArgumentError("poisson_balance_target_scale must be > 0"))
        new(sfs, ams, fs, fk, ds, hs, Float64(poisson_balance_target_scale),
            String(poisson_balance_target_semantics))
    end
end

# Keyword constructor — all fields required.
function TransportBinaryContract(; source_flux_sampling::Symbol,
                                   air_mass_sampling::Symbol,
                                   flux_sampling::Symbol,
                                   flux_kind::Symbol,
                                   delta_semantics::Symbol,
                                   humidity_sampling::Symbol,
                                   poisson_balance_target_scale::Real,
                                   poisson_balance_target_semantics::AbstractString)
    return TransportBinaryContract(source_flux_sampling, air_mass_sampling,
                                   flux_sampling, flux_kind, delta_semantics,
                                   humidity_sampling,
                                   poisson_balance_target_scale,
                                   poisson_balance_target_semantics)
end

"""
    canonical_window_constant_contract(; steps_per_window,
                                         humidity_sampling = :none,
                                         source_flux_sampling = :window_start_endpoint,
                                         include_flux_delta = true) -> TransportBinaryContract

Build the canonical contract for the validated memo-37 path
(`flux_sampling = :window_constant`, per-substep mass amounts). The
Poisson target scale is `1 / (2 * steps_per_window)` — matching the TM5
r1112 horizontal-sweep count of `2 * steps_per_window` per window.

`include_flux_delta = true` implies `delta_semantics =
:forward_window_endpoint_difference` (the writer must include `dm` in the
payload); `false` implies `:none`.
"""
function canonical_window_constant_contract(;
        steps_per_window::Integer,
        humidity_sampling::Symbol = :none,
        source_flux_sampling::Symbol = :window_start_endpoint,
        include_flux_delta::Bool = true)
    return TransportBinaryContract(
        source_flux_sampling = source_flux_sampling,
        air_mass_sampling    = :window_start_endpoint,
        flux_sampling        = :window_constant,
        flux_kind            = :substep_mass_amount,
        delta_semantics      = include_flux_delta ? :forward_window_endpoint_difference : :none,
        humidity_sampling    = humidity_sampling,
        poisson_balance_target_scale = 1.0 / (2 * Int(steps_per_window)),
        poisson_balance_target_semantics = "forward_window_mass_difference / (2 * steps_per_window)",
    )
end

# The cubed-sphere header keys the RUNTIME reads to decide execution cadence and
# capabilities (`uses_binary_substep_contract`, `binary_capabilities`). Emitted
# with defaults by `open_streaming_cs_transport_binary` so they can never go
# missing; listed here once so the writer-side guard and the emitter agree.
const _CS_WRITER_CONTRACT_KEYS = ("runtime_substep_contract",
                                  "preprocessor_contract",
                                  "adaptive_substeps")

"""
    validate_cs_writer_contract!(header::AbstractDict)

Write-time guard: assert every runtime-read cubed-sphere contract key
(`_CS_WRITER_CONTRACT_KEYS`) is present before a binary is finalized — the
writer-side mirror of [`validate_transport_contract!`]. The single choke point
`open_streaming_cs_transport_binary` emits these keys with defaults, so this
never fires in normal use; it exists to fail LOUDLY if a future refactor drops
the default emission, rather than silently shipping a binary that makes the
runtime run convection/chemistry per advection substep (the 2026-05-31 N320
regression: a new source path omitted `runtime_substep_contract`).
"""
function validate_cs_writer_contract!(header::AbstractDict)
    absent = [k for k in _CS_WRITER_CONTRACT_KEYS if !haskey(header, k)]
    isempty(absent) || error(
        "CS transport-binary writer contract violation — runtime-read header " *
        "keys absent: $(join(absent, ", ")). These are emitted with defaults by " *
        "`open_streaming_cs_transport_binary`; a caller or refactor has bypassed " *
        "that single source of truth.")
    return nothing
end

"""
    validate_transport_contract!(header::AbstractDict; allow_legacy::Bool = false)

Assert that `header` declares the current transport-binary contract and that
the timing metadata is self-consistent. `format_version` is a hard boundary:
only `TRANSPORT_BINARY_FORMAT_VERSION` is accepted. Older files are obsolete
and must be regenerated rather than loaded through compatibility defaults.

Shared between `TransportBinaryDriver`, `TransportBinaryReader`, and the
`scripts/diagnostics/inspect_transport_binary.jl` tool so there is ONE
validator every reader-facing tool calls. `allow_legacy` is retained for API
compatibility but no longer bypasses the current runtime contract.
"""
function validate_transport_contract!(header::AbstractDict;
                                      allow_legacy::Bool = false)
    _ = allow_legacy

    missing_or_unknown = String[]
    magic = get(header, "magic", nothing)
    magic == "MFLX" || throw(ArgumentError(
        "Transport-binary contract violation — expected magic=\"MFLX\", got $(repr(magic)). " *
        "This is not a current transport binary."))

    haskey(header, "format_version") || throw(ArgumentError(
        "Transport-binary contract violation — missing format_version. " *
        "All pre-v$(TRANSPORT_BINARY_FORMAT_VERSION) transport binaries are obsolete; regenerate."))
    format_version = try
        Int(header["format_version"])
    catch
        throw(ArgumentError("Transport-binary contract violation — invalid format_version=$(repr(header["format_version"]))"))
    end
    format_version == TRANSPORT_BINARY_FORMAT_VERSION || throw(ArgumentError(
        "Obsolete transport binary format_version=$(format_version); current runtime requires " *
        "format_version=$(TRANSPORT_BINARY_FORMAT_VERSION). Regenerate this file with the current " *
        "preprocessor so the header carries the per-window substep schedule and runtime contract."))

    runtime_contract = get(header, "runtime_substep_contract", nothing)
    if runtime_contract !== nothing
        String(runtime_contract) == "binary_schedule" ||
            throw(ArgumentError("Transport-binary contract violation — unknown runtime_substep_contract=$(repr(runtime_contract))"))
        grid_type = lowercase(String(get(header, "grid_type", "")))
        if grid_type != "cubed_sphere"
            throw(ArgumentError(
                "Transport-binary contract violation — runtime_substep_contract=\"binary_schedule\" " *
                "is currently supported only by CubedSphereTransportDriver. Generic LL/RG binaries " *
                "would otherwise fall back to the runtime CFL pilot and double-subcycle adaptive " *
                "schedules. Add an LL/RG runtime contract before writing adaptive LL/RG binaries."
            ))
        end
    end

    fields = ("source_flux_sampling", "air_mass_sampling", "flux_sampling",
              "flux_kind", "delta_semantics", "humidity_sampling",
              "poisson_balance_target_scale", "poisson_balance_target_semantics",
              "nwindow", "steps_per_window", "steps_per_window_by_window",
              "time_step_schedule", "poisson_balance_target_scale_by_window")

    for f in fields
        if !haskey(header, f)
            push!(missing_or_unknown, "$f (missing)")
        else
            val = header[f]
            if f == "poisson_balance_target_scale"
                # NaN or ≤0 → unknown
                vf = try Float64(val) catch; NaN end
                (isnan(vf) || vf <= 0) && push!(missing_or_unknown, "$f (value=$val)")
            elseif f == "poisson_balance_target_semantics"
                isempty(String(val)) && push!(missing_or_unknown, "$f (empty)")
            elseif f in ("source_flux_sampling", "air_mass_sampling", "flux_sampling",
                         "flux_kind", "delta_semantics", "humidity_sampling")
                sym = _transport_normalize_symbol(val)
                sym === :unknown && push!(missing_or_unknown, "$f (:unknown)")
            end
        end
    end

    if !isempty(missing_or_unknown)
        msg = "Transport-binary contract violation — the following fields are missing " *
              "or unknown in the header:\n  " *
              join(missing_or_unknown, "\n  ") *
              "\nThis binary was produced by a preprocessor that does not declare the " *
              "runtime forcing contract. Regenerate via the current preprocessor " *
              "(scripts/preprocessing/preprocess_transport_binary.jl)."
        throw(ArgumentError(msg))
    end

    # All fields present — validate ranges via a roundtrip construction.
    # This catches e.g. an unknown value for `flux_sampling` that slipped in.
    try
        TransportBinaryContract(
            source_flux_sampling = _transport_normalize_symbol(header["source_flux_sampling"]),
            air_mass_sampling    = _transport_normalize_symbol(header["air_mass_sampling"]),
            flux_sampling        = _transport_normalize_symbol(header["flux_sampling"]),
            flux_kind            = _transport_normalize_symbol(header["flux_kind"]),
            delta_semantics      = _transport_normalize_symbol(header["delta_semantics"]),
            humidity_sampling    = _transport_normalize_symbol(header["humidity_sampling"]),
            poisson_balance_target_scale = Float64(header["poisson_balance_target_scale"]),
            poisson_balance_target_semantics = String(header["poisson_balance_target_semantics"]),
        )
    catch e
        rethrow(e)
    end

    nwindow = try
        Int(header["nwindow"])
    catch
        throw(ArgumentError("Transport-binary contract violation — nwindow must be an integer"))
    end
    nwindow > 0 ||
        throw(ArgumentError("Transport-binary contract violation — nwindow must be positive"))
    steps_per_window = try
        Int(header["steps_per_window"])
    catch
        throw(ArgumentError("Transport-binary contract violation — steps_per_window must be an integer"))
    end
    steps_per_window > 0 ||
        throw(ArgumentError("Transport-binary contract violation — steps_per_window must be positive"))

    schedule = try
        Int.(collect(header["steps_per_window_by_window"]))
    catch
        throw(ArgumentError("Transport-binary contract violation — steps_per_window_by_window must be an integer vector"))
    end
    length(schedule) == nwindow ||
        throw(ArgumentError("Transport-binary contract violation — steps_per_window_by_window length $(length(schedule)) " *
                            "does not match nwindow=$(nwindow)"))
    all(>=(1), schedule) ||
        throw(ArgumentError("Transport-binary contract violation — steps_per_window_by_window must contain only positive integers"))
    steps_per_window == maximum(schedule) ||
        throw(ArgumentError("Transport-binary contract violation — steps_per_window=$(steps_per_window) must equal " *
                            "maximum(steps_per_window_by_window)=$(maximum(schedule))"))

    variable_steps = _has_variable_step_schedule(schedule)
    expected_time_step_schedule = variable_steps ? "per_window" : "constant"
    time_step_schedule = String(header["time_step_schedule"])
    time_step_schedule == expected_time_step_schedule ||
        throw(ArgumentError("Transport-binary contract violation — time_step_schedule=$(repr(time_step_schedule)) " *
                            "but schedule requires $(repr(expected_time_step_schedule))"))

    scalar_scale = Float64(header["poisson_balance_target_scale"])
    expected_scalar_scale = 1.0 / (2 * steps_per_window)
    isapprox(scalar_scale, expected_scalar_scale; atol=eps(Float64) * 8, rtol=0.0) ||
        throw(ArgumentError("Transport-binary contract violation — poisson_balance_target_scale=$(scalar_scale), " *
                            "expected $(expected_scalar_scale) from steps_per_window=$(steps_per_window)"))
    expected_semantics = variable_steps ?
        "forward_window_mass_difference / (2 * steps_per_window_by_window[win])" :
        "forward_window_mass_difference / (2 * steps_per_window)"
    semantics = String(header["poisson_balance_target_semantics"])
    semantics == expected_semantics ||
        throw(ArgumentError("Transport-binary contract violation — poisson_balance_target_semantics=$(repr(semantics)), " *
                            "expected $(repr(expected_semantics))"))

    scale_schedule = try
        Float64.(collect(header["poisson_balance_target_scale_by_window"]))
    catch
        throw(ArgumentError("Transport-binary contract violation — poisson_balance_target_scale_by_window must be a numeric vector"))
    end
    length(scale_schedule) == nwindow ||
        throw(ArgumentError("Transport-binary contract violation — poisson_balance_target_scale_by_window length $(length(scale_schedule)) " *
                            "does not match nwindow=$(nwindow)"))
    for win in 1:nwindow
        expected = 1.0 / (2 * schedule[win])
        isapprox(scale_schedule[win], expected; atol=eps(Float64) * 8, rtol=0.0) ||
            throw(ArgumentError("Transport-binary contract violation — poisson_balance_target_scale_by_window[$win]=" *
                                "$(scale_schedule[win]), expected $(expected) from steps_per_window_by_window[$win]=$(schedule[win])"))
    end
    return nothing
end
