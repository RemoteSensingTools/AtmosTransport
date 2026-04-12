# ---------------------------------------------------------------------------
# TransportPolicy — single source of truth for transport configuration
#
# Resolves all transport-related flags into one immutable policy object.
# Passed through the run loop to ensure consistent behavior across all
# physics phases. Replaces scattered boolean flags (vertical_remap,
# dry_correction, mass_fixer) with a single typed object.
#
# See docs/src/developer/TRANSPORT_COMPARISON.md §4 for design rationale.
# ---------------------------------------------------------------------------

"""Transport basis for multiple dispatch in the GCHP advection path."""
abstract type AbstractTransportBasis end

"""Dry-air transport basis: dp_dry = DELP×(1-QV), dry MFX, direct-cumsum PE."""
struct DryTransportBasis <: AbstractTransportBasis end

"""Moist-air transport basis (GCHP-faithful): moist dp/MFX, hybrid PE, prognostic QV."""
struct MoistTransportBasis <: AbstractTransportBasis end

"""
$(TYPEDEF)

Central transport configuration resolved once per simulation run.

All transport-related options (vertical method, pressure basis, mass balance)
are collected here so that the run loop and physics phases can dispatch on
policy fields rather than querying scattered metadata flags.

# Construction

Use `resolve_transport_policy(config)` to build from TOML config, or construct
directly for programmatic use:

    TransportPolicy(vertical_operator=:pressure_remap, pressure_basis=:dry)

# Fields

$(FIELDS)
"""
struct TransportPolicy
    "Vertical transport method: `:continuity_cm` (Strang Z-advection) or `:pressure_remap` (FV3-style PPM remap)"
    vertical_operator::Symbol
    "Air mass basis for advection: `:dry` (MFXC/MFYC are dry) or `:moist`"
    pressure_basis::Symbol
    "Mass balance correction: `:none`, `:column`, or `:global_fixer`"
    mass_balance_mode::Symbol
    "PE computation for vertical remap: `:direct_cumsum` (default) or `:hybrid` (GCHP-style ak+bk×PS + calcScalingFactor)"
    pe_method::Symbol

    function TransportPolicy(; vertical_operator::Symbol = :continuity_cm,
                               pressure_basis::Symbol = :dry,
                               mass_balance_mode::Symbol = :global_fixer,
                               pe_method::Symbol = :direct_cumsum)
        vertical_operator ∈ (:continuity_cm, :pressure_remap) ||
            throw(ArgumentError("vertical_operator must be :continuity_cm or :pressure_remap, got :$vertical_operator"))
        pressure_basis ∈ (:dry, :moist) ||
            throw(ArgumentError("pressure_basis must be :dry or :moist, got :$pressure_basis"))
        mass_balance_mode ∈ (:none, :column, :global_fixer) ||
            throw(ArgumentError("mass_balance_mode must be :none, :column, or :global_fixer, got :$mass_balance_mode"))
        pe_method ∈ (:direct_cumsum, :hybrid) ||
            throw(ArgumentError("pe_method must be :direct_cumsum or :hybrid, got :$pe_method"))
        new(vertical_operator, pressure_basis, mass_balance_mode, pe_method)
    end
end

"""Return the transport basis type from a TransportPolicy."""
transport_basis_type(p::TransportPolicy) = p.pressure_basis === :moist ? MoistTransportBasis() : DryTransportBasis()

"""
    resolve_transport_policy(metadata::Dict) → TransportPolicy

Resolve transport policy from model metadata (parsed from TOML config).
Maps legacy boolean flags to policy fields for backward compatibility.
"""
function resolve_transport_policy(metadata::Dict)
    # Legacy flag mapping
    vertical = get(metadata, "vertical_remap", false) ? :pressure_remap : :continuity_cm
    basis    = get(metadata, "dry_correction", false) ? :dry : :moist
    fixer    = get(metadata, "mass_fixer", false)     ? :global_fixer : :none
    pe       = Symbol(get(metadata, "pe_method", "direct_cumsum"))

    # Explicit policy fields override legacy flags
    vertical = Symbol(get(metadata, "vertical_operator", string(vertical)))
    basis    = Symbol(get(metadata, "pressure_basis", string(basis)))
    fixer    = Symbol(get(metadata, "mass_balance_mode", string(fixer)))
    pe       = Symbol(get(metadata, "pe_method", string(pe)))

    policy = TransportPolicy(; vertical_operator=vertical, pressure_basis=basis,
                               mass_balance_mode=fixer, pe_method=pe)

    @info "Transport policy resolved" policy.vertical_operator policy.pressure_basis policy.mass_balance_mode policy.pe_method

    return policy
end

function Base.show(io::IO, p::TransportPolicy)
    print(io, "TransportPolicy(vertical=$(p.vertical_operator), basis=$(p.pressure_basis), mass_balance=$(p.mass_balance_mode), pe=$(p.pe_method))")
end
