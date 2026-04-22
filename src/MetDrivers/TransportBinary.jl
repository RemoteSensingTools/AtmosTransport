using Mmap
using JSON3
using Base.Threads
using ..Architectures: CPU
import ..State: mass_basis

"""
    TransportBinaryHeader

Metadata for a topology-generic preprocessed transport binary.

## Key fields

- `grid_type`: `:latlon` or `:reduced_gaussian`
- `horizontal_topology`: `:structureddirectional` (LL, arrays `(Nx, Ny, Nz)`)
  or `:faceindexed` (RG, arrays `(ncell, Nz)` + `(nface_h, Nz)`)
- `ncell`: total horizontal cells (LL: `Nx × Ny`; RG: sum of nlon_per_ring)
- `nface_h`: total horizontal faces (LL: `(Nx+1)×Ny + Nx×(Ny+1)`;
  RG: `ncell + Σ boundary_counts`)
- `nlevel`: number of vertical levels (k=1 = TOA, k=nlevel = surface)
- `nwindow`: windows per day (typically 24 for hourly ERA5)
- `A_ifc`: hybrid A coefficients [Pa] at `nlevel + 1` half-levels.
  `p_half[k] = A_ifc[k] + B_ifc[k] × ps`. For ERA5 tropo34: `A_ifc[1] = 0`
  (TOA), `A_ifc[end] = 0` (surface where B=1).
- `B_ifc`: hybrid B coefficients [dimensionless] at `nlevel + 1` half-levels.
  `B_ifc[1] = 0` (TOA, pure pressure levels), `B_ifc[end] = 1.0` (surface,
  pure sigma level).
- `mass_basis`: `:moist` or `:dry` — determines whether the stored air mass
  includes or excludes water vapour.
- `flux_kind`: `:substep_mass_amount` — stored flux is the mass [kg] per
  substep (divide by `steps_per_window` to get per-window total).
- `flux_sampling`: `:window_constant` — same flux applied at every substep
  within a window.
"""
struct TransportBinaryHeader
    format_version       :: Int
    header_bytes         :: Int
    on_disk_float_type   :: Symbol      # :Float32 or :Float64
    float_bytes          :: Int         # 4 or 8
    grid_type            :: Symbol      # :latlon or :reduced_gaussian
    horizontal_topology  :: Symbol      # :structureddirectional or :faceindexed
    ncell                :: Int         # total horizontal cells
    nface_h              :: Int         # total horizontal faces
    nlevel               :: Int         # vertical levels (k=1 TOA, k=nlevel surface)
    nwindow              :: Int         # windows per day (typically 24)
    dt_met_seconds       :: Float64     # met-data window interval [s] (3600 for hourly)
    half_dt_seconds      :: Float64     # half-step time [s] for flux scaling
    steps_per_window     :: Int         # substeps per window (window_dt / dt)
    source_flux_sampling :: Symbol
    air_mass_sampling    :: Symbol
    flux_sampling        :: Symbol      # :window_constant
    flux_kind            :: Symbol      # :substep_mass_amount
    humidity_sampling    :: Symbol
    delta_semantics      :: Symbol
    poisson_balance_target_scale :: Float64
    poisson_balance_target_semantics :: String
    A_ifc                :: Vector{Float64}  # hybrid A [Pa], length nlevel+1
    B_ifc                :: Vector{Float64}  # hybrid B [1],  length nlevel+1
    mass_basis           :: Symbol           # :moist or :dry
    payload_sections     :: Vector{Symbol}
    include_qv           :: Bool
    include_qv_endpoints :: Bool
    n_qv                 :: Int
    n_qv_start           :: Int
    n_qv_end             :: Int
    n_geometry_elems     :: Int
    elems_per_window     :: Int
    Nx                   :: Int
    Ny                   :: Int
    lons_f64             :: Vector{Float64}
    lats_f64             :: Vector{Float64}
    longitude_interval_f64 :: Vector{Float64}
    latitude_interval_f64  :: Vector{Float64}
    nlat                 :: Int
    ring_latitudes_f64   :: Vector{Float64}
    nlon_per_ring        :: Vector{Int}
end

"""
    TransportBinaryReader{FT, DiskFT}

Reader for topology-generic preprocessed transport binaries.

Implemented combinations:
- `grid_type = :latlon`, `horizontal_topology = :structureddirectional`
- `grid_type = :reduced_gaussian`, `horizontal_topology = :faceindexed`
"""
struct TransportBinaryReader{FT, DiskFT}
    data   :: Vector{DiskFT}
    io     :: IOStream
    header :: TransportBinaryHeader
    path   :: String
end

@inline function _transport_geometry_summary(h::TransportBinaryHeader)
    if _transport_is_structured(h)
        return string(h.Nx, "×", h.Ny, " structured cells, ", h.nlevel, " levels")
    elseif _transport_is_faceindexed(h)
        return string(h.ncell, " cells, ", h.nface_h, " faces, ", h.nlevel, " levels")
    else
        return string(h.ncell, " cells, ", h.nlevel, " levels")
    end
end

@inline function _transport_qv_summary(h::TransportBinaryHeader)
    if h.include_qv_endpoints
        return "qv_start/qv_end"
    elseif h.include_qv
        return "qv"
    else
        return "none"
    end
end

@inline function _transport_semantics_summary(h::TransportBinaryHeader)
    return string(
        "air_mass=", h.air_mass_sampling,
        ", flux=", h.flux_sampling, "/", h.flux_kind,
        ", humidity=", h.humidity_sampling,
        ", delta=", h.delta_semantics,
        h.source_flux_sampling === :unknown ? "" : string(", source_flux=", h.source_flux_sampling)
    )
end

function Base.summary(h::TransportBinaryHeader)
    return string(
        "TransportBinaryHeader(v", h.format_version, ", ",
        h.grid_type, "/", h.horizontal_topology, ", ",
        h.nwindow, " windows)"
    )
end

function Base.show(io::IO, h::TransportBinaryHeader)
    print(io, summary(h), "\n",
          "├── geometry:      ", _transport_geometry_summary(h), "\n",
          "├── storage:       ", h.on_disk_float_type, " on disk, basis=", h.mass_basis, "\n",
          "├── timing:        dt=", h.dt_met_seconds, " s, steps/window=", h.steps_per_window, "\n",
          "├── payload:       ", join(String.(h.payload_sections), ", "), "\n",
          "├── humidity:      ", _transport_qv_summary(h), "\n",
          "├── semantics:     ", _transport_semantics_summary(h), "\n",
          "├── poisson:       ", isnan(h.poisson_balance_target_scale) ? "unspecified" :
                               string("scale=", h.poisson_balance_target_scale, ", ", h.poisson_balance_target_semantics), "\n",
          "└── header bytes:  ", h.header_bytes)
end

function Base.summary(r::TransportBinaryReader{FT, DiskFT}) where {FT, DiskFT}
    return string(
        "TransportBinaryReader{", FT, "←", DiskFT, "}(",
        basename(r.path), ", ", r.header.grid_type, "/", r.header.horizontal_topology, ", ",
        r.header.nwindow, " windows)"
    )
end

function Base.show(io::IO, r::TransportBinaryReader)
    h = r.header
    print(io, summary(r), "\n",
          "├── path:          ", r.path, "\n",
          "├── geometry:      ", _transport_geometry_summary(h), "\n",
          "├── storage:       ", h.on_disk_float_type, " on disk, load as ", eltype(r.data), "\n",
          "├── basis:         ", h.mass_basis, "\n",
          "├── timing:        dt=", h.dt_met_seconds, " s, steps/window=", h.steps_per_window, "\n",
          "├── payload:       ", join(String.(h.payload_sections), ", "), "\n",
          "├── humidity:      ", _transport_qv_summary(h), "\n",
          "├── semantics:     ", _transport_semantics_summary(h), "\n",
          "├── poisson:       ", isnan(h.poisson_balance_target_scale) ? "unspecified" :
                               string("scale=", h.poisson_balance_target_scale, ", ", h.poisson_balance_target_semantics), "\n",
          "└── windows:       ", h.nwindow)
end

window_count(r::TransportBinaryReader) = r.header.nwindow
mass_basis(r::TransportBinaryReader) = r.header.mass_basis
grid_type(r::TransportBinaryReader) = r.header.grid_type
horizontal_topology(r::TransportBinaryReader) = r.header.horizontal_topology
source_flux_sampling(r::TransportBinaryReader) = r.header.source_flux_sampling
air_mass_sampling(r::TransportBinaryReader) = r.header.air_mass_sampling
flux_sampling(r::TransportBinaryReader) = r.header.flux_sampling
flux_kind(r::TransportBinaryReader) = r.header.flux_kind
humidity_sampling(r::TransportBinaryReader) = r.header.humidity_sampling
delta_semantics(r::TransportBinaryReader) = r.header.delta_semantics
A_ifc(r::TransportBinaryReader) = r.header.A_ifc
B_ifc(r::TransportBinaryReader) = r.header.B_ifc
has_qv(r::TransportBinaryReader) = r.header.include_qv || r.header.include_qv_endpoints
has_qv_endpoints(r::TransportBinaryReader) = r.header.include_qv_endpoints
has_flux_delta(r::TransportBinaryReader) = any(section in (:dam, :dbm, :dcm, :dm, :dhflux) for section in r.header.payload_sections)

"""
    has_tm5_convection(r::TransportBinaryReader) -> Bool

`true` if the binary carries all four TM5 convection sections
(`entu`, `detu`, `entd`, `detd`) — the contract enforced by the
preprocessor when `tm5_convection = true`. Used by the
`TransportBinaryDriver` to decide whether to populate
`ConvectionForcing.tm5_fields` on loaded windows.
"""
has_tm5_convection(r::TransportBinaryReader) =
    all(s in r.header.payload_sections for s in (:entu, :detu, :entd, :detd))

_transport_is_structured(h::TransportBinaryHeader) =
    h.grid_type === :latlon && h.horizontal_topology === :structureddirectional
_transport_is_faceindexed(h::TransportBinaryHeader) =
    h.grid_type === :reduced_gaussian && h.horizontal_topology === :faceindexed

function _transport_disk_float_type(sym::Symbol)
    sym === :Float64 ? Float64 : Float32
end

function _transport_parse_on_disk_float_type(hdr)
    ft_str = string(get(hdr, :float_type, "Float32"))
    if ft_str == "Float64"
        return :Float64, 8
    else
        return :Float32, 4
    end
end

function _transport_parse_mass_basis(hdr)
    if !haskey(hdr, :mass_basis)
        @warn "Transport binary header has no mass_basis field — assuming moist (legacy binary). " *
              "Regenerate with the current preprocessor for dry-basis binaries (Invariant 14)."
    end
    basis_str = lowercase(string(get(hdr, :mass_basis, "moist")))
    return basis_str == "dry" ? :dry : :moist
end

@inline function _transport_normalize_symbol(value)
    return Symbol(replace(lowercase(String(value)), '-' => '_', ' ' => '_'))
end

const _TRANSPORT_ALLOWED_SOURCE_FLUX_SAMPLINGS = (
    :window_start_endpoint,
    :window_end_endpoint,
    :window_mean,
    :interval_integrated,
)

@inline function _transport_validate_source_flux_sampling(value)
    sym = _transport_normalize_symbol(value)
    sym in _TRANSPORT_ALLOWED_SOURCE_FLUX_SAMPLINGS || throw(ArgumentError(
        "unsupported source_flux_sampling=$(value); supported values are " *
        join(string.(Tuple(_TRANSPORT_ALLOWED_SOURCE_FLUX_SAMPLINGS)), ", ")
    ))
    return sym
end

@inline function _transport_parse_symbol_key(hdr, key::Symbol, default::Symbol)
    return _transport_normalize_symbol(get(hdr, key, String(default)))
end

@inline function _transport_default_humidity_sampling(payload_sections::AbstractVector{Symbol})
    if (:qv_start in payload_sections) || (:qv_end in payload_sections)
        return :window_endpoints
    elseif :qv in payload_sections
        return :single_field
    else
        return :none
    end
end

@inline function _transport_default_delta_semantics(payload_sections::AbstractVector{Symbol})
    return any(section in (:dam, :dbm, :dcm, :dm, :dhflux) for section in payload_sections) ?
           :forward_window_endpoint_difference : :none
end

# ===========================================================================
# TransportBinaryContract — self-describing timing/basis semantics.
#
# Every writer must supply an explicit contract; every reader must validate
# one. Silent defaults are how plan 24 Commit 4's LL+TM5 binary landed on
# disk without declaring `flux_sampling=:window_constant`, the runtime
# parser defaulted to `:window_start_endpoint`, and the runtime ran the
# pre-memo-37 bug class. See docs/37_WINDOW_CONSTANT_FLUX_INTERPRETATION_BUG.
# ===========================================================================

const _TRANSPORT_ALLOWED_AIR_MASS_SAMPLINGS = (:window_start_endpoint,)
const _TRANSPORT_ALLOWED_FLUX_SAMPLINGS     = (:window_start_endpoint, :window_constant, :window_mean)
const _TRANSPORT_ALLOWED_FLUX_KINDS         = (:substep_mass_amount,)
const _TRANSPORT_ALLOWED_DELTA_SEMANTICS    = (:forward_window_endpoint_difference, :none)
const _TRANSPORT_ALLOWED_HUMIDITY_SAMPLINGS = (:window_endpoints, :single_field, :none)

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

"""
    validate_transport_contract!(header::AbstractDict; allow_legacy::Bool = false)

Assert that `header` declares all eight contract fields and that they are
self-consistent. Throws `ArgumentError` with a clear, action-named message
when a field is missing or unknown. When `allow_legacy=true` (or when the
environment variable `ATMOSTR_ALLOW_LEGACY_BINARY` is set to `"1"`),
missing/unknown fields are demoted to `@warn` and a "this binary cannot be
trusted" banner, and the function returns a (best-effort) contract filled
with `:unknown` placeholders.

Shared between `TransportBinaryDriver`, `TransportBinaryReader`, and the
`scripts/diagnostics/inspect_transport_binary.jl` tool so there is ONE
validator every reader-facing tool calls. Wiring these three call sites
is deferred to Commit D; this commit just introduces the function.
"""
function validate_transport_contract!(header::AbstractDict;
                                      allow_legacy::Bool = false)
    envvar = get(ENV, "ATMOSTR_ALLOW_LEGACY_BINARY", "")
    effective_allow = allow_legacy || envvar == "1"

    missing_or_unknown = String[]
    fields = ("source_flux_sampling", "air_mass_sampling", "flux_sampling",
              "flux_kind", "delta_semantics", "humidity_sampling",
              "poisson_balance_target_scale", "poisson_balance_target_semantics")

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
            else
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
              "(scripts/preprocessing/preprocess_transport_binary.jl) or, if you MUST " *
              "load a legacy binary for inspection only, set " *
              "ATMOSTR_ALLOW_LEGACY_BINARY=1 (the runtime will not guarantee correctness)."
        if effective_allow
            @warn "╔══ LEGACY TRANSPORT BINARY — contract violations ignored ══╗\n" * msg *
                  "\n╚═══════════════════════════════════════════════════════════╝"
            return nothing
        else
            throw(ArgumentError(msg))
        end
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
        if effective_allow
            @warn "Transport-binary contract values out of range (legacy bypass on): $e"
        else
            rethrow(e)
        end
    end
    return nothing
end

_transport_parse_grid_type(hdr) = Symbol(lowercase(string(hdr.grid_type)))
_transport_parse_topology(hdr) = Symbol(lowercase(string(hdr.horizontal_topology)))

function _transport_parse_sections(hdr)
    sections = haskey(hdr, :payload_sections) ? collect(hdr.payload_sections) : ["m", "hflux", "cm", "ps"]
    return Symbol.(lowercase.(String.(sections)))
end

function _transport_header_axis(hdr, n::Int, key1::Symbol, key2::Symbol)
    if haskey(hdr, key1)
        return Float64.(collect(getproperty(hdr, key1)))
    elseif haskey(hdr, key2)
        return Float64.(collect(getproperty(hdr, key2)))
    elseif n == 0
        return Float64[]
    else
        error("Transport binary header missing $(key1) / $(key2)")
    end
end

function _transport_header_interval(hdr, key1::Symbol, key2::Symbol)
    if haskey(hdr, key1)
        return Float64.(collect(getproperty(hdr, key1)))
    elseif haskey(hdr, key2)
        return Float64.(collect(getproperty(hdr, key2)))
    else
        return Float64[]
    end
end

function _parse_transport_header(raw_bytes::Vector{UInt8})
    json_end = something(findfirst(==(0x00), raw_bytes), length(raw_bytes) + 1) - 1
    hdr = JSON3.read(String(raw_bytes[1:json_end]))

    haskey(hdr, :format_version) ||
        error("TransportBinaryReader requires the topology-generic binary family header (`format_version` missing)")

    # Plan 39 Commit D: no more silent defaults for missing contract fields.
    # `validate_transport_contract!` (called by `TransportBinaryReader`
    # before we get here) has already verified the 8 fields are present —
    # unless the env-var legacy bypass was set, in which case missing
    # fields come back as :unknown and downstream code must handle them
    # (or trip on a later typed check).
    format_version = Int(hdr.format_version)
    header_bytes = Int(get(hdr, :header_bytes, 16384))
    disk_ft, float_bytes = _transport_parse_on_disk_float_type(hdr)
    grid_type = _transport_parse_grid_type(hdr)
    topology = _transport_parse_topology(hdr)
    ncell = Int(hdr.ncell)
    nface_h = Int(hdr.nface_h)
    nlevel = Int(hdr.nlevel)
    nwindow = Int(hdr.nwindow)
    A_ifc = Float64.(collect(hdr.A_ifc))
    B_ifc = Float64.(collect(hdr.B_ifc))
    payload_sections = _transport_parse_sections(hdr)
    include_qv = Bool(get(hdr, :include_qv, false))
    include_qv_endpoints = Bool(get(hdr, :include_qv_endpoints, false))
    source_flux_sampling = _transport_parse_symbol_key(hdr, :source_flux_sampling, :unknown)
    air_mass_sampling    = _transport_parse_symbol_key(hdr, :air_mass_sampling,    :unknown)
    flux_sampling        = _transport_parse_symbol_key(hdr, :flux_sampling,        :unknown)
    flux_kind            = _transport_parse_symbol_key(hdr, :flux_kind,            :unknown)
    humidity_sampling    = _transport_parse_symbol_key(hdr, :humidity_sampling,    :unknown)
    delta_semantics      = _transport_parse_symbol_key(hdr, :delta_semantics,      :unknown)
    poisson_balance_target_scale = haskey(hdr, :poisson_balance_target_scale) ?
                                   Float64(hdr.poisson_balance_target_scale) : NaN
    poisson_balance_target_semantics = haskey(hdr, :poisson_balance_target_semantics) ?
                                       String(hdr.poisson_balance_target_semantics) : ""
    n_qv = Int(get(hdr, :n_qv, include_qv ? ncell * nlevel : 0))
    n_qv_start = Int(get(hdr, :n_qv_start, include_qv_endpoints ? ncell * nlevel : 0))
    n_qv_end = Int(get(hdr, :n_qv_end, include_qv_endpoints ? ncell * nlevel : 0))
    n_geometry_elems = Int(get(hdr, :n_geometry_elems, 0))
    elems_per_window = Int(hdr.elems_per_window)

    Nx = haskey(hdr, :Nx) ? Int(hdr.Nx) : 0
    Ny = haskey(hdr, :Ny) ? Int(hdr.Ny) : 0
    lons_f64 = _transport_header_axis(hdr, Nx, :lons, :lon_centers)
    lats_f64 = _transport_header_axis(hdr, Ny, :lats, :lat_centers)
    longitude_interval_f64 = _transport_header_interval(hdr, :longitude_interval, :longitude)
    latitude_interval_f64 = _transport_header_interval(hdr, :latitude_interval, :latitude)

    nlon_per_ring = haskey(hdr, :nlon_per_ring) ? Int.(collect(hdr.nlon_per_ring)) : Int[]
    ring_latitudes = if haskey(hdr, :latitudes)
        Float64.(collect(hdr.latitudes))
    elseif haskey(hdr, :ring_latitudes)
        Float64.(collect(hdr.ring_latitudes))
    else
        Float64[]
    end
    nlat = haskey(hdr, :nlat) ? Int(hdr.nlat) : length(ring_latitudes)

    return TransportBinaryHeader(
        format_version,
        header_bytes,
        disk_ft,
        float_bytes,
        grid_type,
        topology,
        ncell,
        nface_h,
        nlevel,
        nwindow,
        Float64(hdr.dt_met_seconds),
        Float64(hdr.half_dt_seconds),
        Int(hdr.steps_per_window),
        source_flux_sampling,
        air_mass_sampling,
        flux_sampling,
        flux_kind,
        humidity_sampling,
        delta_semantics,
        poisson_balance_target_scale,
        poisson_balance_target_semantics,
        A_ifc,
        B_ifc,
        _transport_parse_mass_basis(hdr),
        payload_sections,
        include_qv,
        include_qv_endpoints,
        n_qv,
        n_qv_start,
        n_qv_end,
        n_geometry_elems,
        elems_per_window,
        Nx,
        Ny,
        lons_f64,
        lats_f64,
        longitude_interval_f64,
        latitude_interval_f64,
        nlat,
        ring_latitudes,
        nlon_per_ring,
    )
end

function TransportBinaryReader(bin_path::String; FT::Type{<:AbstractFloat} = Float32)
    io = open(bin_path, "r")
    read_sz = min(262144, filesize(bin_path))
    raw = read(io, read_sz)

    # Plan 39 Commit D: validate the self-describing transport-binary
    # contract BEFORE mmap'ing the payload. Rejects ambiguous/legacy
    # headers with a clear error (names the missing field + regeneration
    # command). Env-var ATMOSTR_ALLOW_LEGACY_BINARY=1 demotes to loud warn
    # for inspection-only loads — downstream semantics are then :unknown.
    # This call site is shared with `TransportBinaryDriver` and the
    # `scripts/diagnostics/inspect_transport_binary.jl` tool, so ONE
    # validator gates every reader-facing entry point.
    json_end = something(findfirst(==(0x00), raw), length(raw) + 1) - 1
    hdr_dict = try
        hdr_obj = JSON3.read(String(raw[1:json_end]))
        # Convert JSON3.Object to a plain Dict for the validator.
        Dict{String, Any}(String(k) => v for (k, v) in pairs(hdr_obj))
    catch e
        close(io)
        rethrow(e)
    end
    try
        validate_transport_contract!(hdr_dict)
    catch e
        close(io)
        rethrow(e)
    end

    header = _parse_transport_header(raw)

    DiskFT = _transport_disk_float_type(header.on_disk_float_type)
    total_elems = header.n_geometry_elems + header.elems_per_window * header.nwindow
    seek(io, header.header_bytes)
    data = Mmap.mmap(io, Vector{DiskFT}, total_elems, header.header_bytes)

    return TransportBinaryReader{FT, DiskFT}(data, io, header, bin_path)
end

Base.close(r::TransportBinaryReader) = close(r.io)

@inline _transport_window_offset(r::TransportBinaryReader, win::Int) =
    r.header.n_geometry_elems + (win - 1) * r.header.elems_per_window

@inline function _transport_structured_section_elements(Nx::Int, Ny::Int, ncell::Int, nlevel::Int, section::Symbol)
    if section === :m || section === :dm || section === :qv || section === :qv_start || section === :qv_end
        return ncell * nlevel
    elseif section === :am || section === :dam
        return (Nx + 1) * Ny * nlevel
    elseif section === :bm || section === :dbm
        return Nx * (Ny + 1) * nlevel
    elseif section === :cm || section === :dcm
        return ncell * (nlevel + 1)
    elseif section === :ps
        return ncell
    # TM5 convection (plan 23 Commit 3): four layer-center fields.
    elseif section === :entu || section === :detu ||
           section === :entd || section === :detd
        return ncell * nlevel
    else
        error("Unsupported structured payload section: $section")
    end
end

@inline function _transport_faceindexed_section_elements(ncell::Int, nface_h::Int, nlevel::Int, section::Symbol)
    if section === :m || section === :dm || section === :qv || section === :qv_start || section === :qv_end
        return ncell * nlevel
    elseif section === :hflux || section === :dhflux
        return nface_h * nlevel
    elseif section === :cm || section === :dcm
        return ncell * (nlevel + 1)
    elseif section === :ps
        return ncell
    # TM5 convection (plan 23 Commit 3): four layer-center fields.
    elseif section === :entu || section === :detu ||
           section === :entd || section === :detd
        return ncell * nlevel
    else
        error("Unsupported face-indexed payload section: $section")
    end
end

function _transport_section_elements(h::TransportBinaryHeader, section::Symbol)
    if _transport_is_structured(h)
        return _transport_structured_section_elements(h.Nx, h.Ny, h.ncell, h.nlevel, section)
    elseif _transport_is_faceindexed(h)
        return _transport_faceindexed_section_elements(h.ncell, h.nface_h, h.nlevel, section)
    else
        error("Unsupported payload section $(section) for grid/topology $(h.grid_type) / $(h.horizontal_topology)")
    end
end

_transport_basis_symbol(sym::Symbol) = lowercase(String(sym)) == "dry" ? :dry : :moist
_transport_basis_symbol(::DryBasis) = :dry
_transport_basis_symbol(::MoistBasis) = :moist

_transport_window_mass(window) =
    haskey(window, :m) ? window.m :
    haskey(window, :state) ? window.state.air_mass :
    error("transport-binary window is missing `m` or `state`")

_transport_window_ps(window) =
    haskey(window, :ps) ? window.ps :
    error("transport-binary window is missing `ps`")

_transport_window_hflux(window) =
    haskey(window, :hflux) ? window.hflux :
    haskey(window, :fluxes) ? window.fluxes.horizontal_flux :
    error("transport-binary window is missing `hflux` or `fluxes`")

_transport_window_am(window) =
    haskey(window, :am) ? window.am :
    haskey(window, :fluxes) ? window.fluxes.am :
    error("transport-binary window is missing `am` or `fluxes`")

_transport_window_bm(window) =
    haskey(window, :bm) ? window.bm :
    haskey(window, :fluxes) ? window.fluxes.bm :
    error("transport-binary window is missing `bm` or `fluxes`")

_transport_window_cm(window) =
    haskey(window, :cm) ? window.cm :
    haskey(window, :fluxes) ? window.fluxes.cm :
    error("transport-binary window is missing `cm` or `fluxes`")

_transport_window_dam(window) = haskey(window, :dam) ? window.dam : error("transport-binary window is missing `dam`")
_transport_window_dbm(window) = haskey(window, :dbm) ? window.dbm : error("transport-binary window is missing `dbm`")
_transport_window_dhflux(window) = haskey(window, :dhflux) ? window.dhflux : error("transport-binary window is missing `dhflux`")
_transport_window_dcm(window) = haskey(window, :dcm) ? window.dcm : error("transport-binary window is missing `dcm`")
_transport_window_dm(window) = haskey(window, :dm) ? window.dm : error("transport-binary window is missing `dm`")

# TM5 convection payload writers read a NamedTuple of four
# layer-center fields from the preprocessor window.  The
# preprocessor supplies `window.tm5_fields.entu`, `.detu`, `.entd`,
# `.detd`.  Errors loudly if the writer requested a TM5 section
# but the window didn't include `tm5_fields` (plan 23 Commit 3).
@inline function _transport_window_tm5_field(window, name::Symbol)
    haskey(window, :tm5_fields) ||
        error("transport-binary window is missing `tm5_fields` " *
              "(required when writing TM5 convection sections)")
    nt = window.tm5_fields
    hasproperty(nt, name) ||
        error("transport-binary window.tm5_fields is missing field `$(name)`")
    return getproperty(nt, name)
end

function _transport_window_field(window, section::Symbol)
    if section === :m
        return _transport_window_mass(window)
    elseif section === :am
        return _transport_window_am(window)
    elseif section === :bm
        return _transport_window_bm(window)
    elseif section === :hflux
        return _transport_window_hflux(window)
    elseif section === :cm
        return _transport_window_cm(window)
    elseif section === :dam
        return _transport_window_dam(window)
    elseif section === :dbm
        return _transport_window_dbm(window)
    elseif section === :dhflux
        return _transport_window_dhflux(window)
    elseif section === :dcm
        return _transport_window_dcm(window)
    elseif section === :dm
        return _transport_window_dm(window)
    elseif section === :ps
        return _transport_window_ps(window)
    elseif section === :qv
        return window.qv
    elseif section === :qv_start
        return window.qv_start
    elseif section === :qv_end
        return window.qv_end
    # TM5 convection fields (plan 23 Commit 3).
    elseif section === :entu
        return _transport_window_tm5_field(window, :entu)
    elseif section === :detu
        return _transport_window_tm5_field(window, :detu)
    elseif section === :entd
        return _transport_window_tm5_field(window, :entd)
    elseif section === :detd
        return _transport_window_tm5_field(window, :detd)
    else
        error("Unsupported transport-binary section: $section")
    end
end

function _transport_push_optional_sections!(sections::Vector{Symbol}, window)
    haskey(window, :qv) && push!(sections, :qv)
    haskey(window, :qv_start) && push!(sections, :qv_start)
    haskey(window, :qv_end) && push!(sections, :qv_end)
    haskey(window, :dam) && push!(sections, :dam)
    haskey(window, :dbm) && push!(sections, :dbm)
    haskey(window, :dhflux) && push!(sections, :dhflux)
    haskey(window, :dcm) && push!(sections, :dcm)
    haskey(window, :dm) && push!(sections, :dm)
    # Plan 23 Commit 3: TM5 convection adds a NamedTuple of four
    # layer-center fields.  Writer emits these when the preprocessor
    # window provides `tm5_fields`; reader populates
    # `ConvectionForcing.tm5_fields` from the corresponding binary
    # sections.
    if haskey(window, :tm5_fields) && window.tm5_fields !== nothing
        push!(sections, :entu)
        push!(sections, :detu)
        push!(sections, :entd)
        push!(sections, :detd)
    end
    return sections
end

function _transport_payload_sections(::AtmosGrid{<:LatLonMesh}, window)
    return _transport_push_optional_sections!(Symbol[:m, :am, :bm, :cm, :ps], window)
end

function _transport_payload_sections(::AtmosGrid{<:ReducedGaussianMesh}, window)
    return _transport_push_optional_sections!(Symbol[:m, :hflux, :cm, :ps], window)
end

function _transport_validate_basis(window, basis_sym::Symbol)
    if haskey(window, :state)
        _transport_basis_symbol(mass_basis(window.state)) == basis_sym ||
            throw(ArgumentError("window state basis does not match requested transport binary basis $(basis_sym)"))
    end
    if haskey(window, :fluxes)
        _transport_basis_symbol(mass_basis(window.fluxes)) == basis_sym ||
            throw(ArgumentError("window flux basis does not match requested transport binary basis $(basis_sym)"))
    end
    return nothing
end

function _transport_validate_optional_qv(window, expected)
    if haskey(window, :qv)
        size(window.qv) == expected ||
            throw(DimensionMismatch("window qv has size $(size(window.qv)), expected $(expected)"))
    end
    if haskey(window, :qv_start)
        size(window.qv_start) == expected ||
            throw(DimensionMismatch("window qv_start has size $(size(window.qv_start)), expected $(expected)"))
    end
    if haskey(window, :qv_end)
        size(window.qv_end) == expected ||
            throw(DimensionMismatch("window qv_end has size $(size(window.qv_end)), expected $(expected)"))
    end
    return nothing
end

function _transport_validate_optional_structured_deltas(window, Nx::Int, Ny::Int, nlevel::Int)
    if haskey(window, :dam)
        size(window.dam) == (Nx + 1, Ny, nlevel) ||
            throw(DimensionMismatch("window dam has size $(size(window.dam)), expected $((Nx + 1, Ny, nlevel))"))
    end
    if haskey(window, :dbm)
        size(window.dbm) == (Nx, Ny + 1, nlevel) ||
            throw(DimensionMismatch("window dbm has size $(size(window.dbm)), expected $((Nx, Ny + 1, nlevel))"))
    end
    if haskey(window, :dcm)
        size(window.dcm) == (Nx, Ny, nlevel + 1) ||
            throw(DimensionMismatch("window dcm has size $(size(window.dcm)), expected $((Nx, Ny, nlevel + 1))"))
    end
    if haskey(window, :dm)
        size(window.dm) == (Nx, Ny, nlevel) ||
            throw(DimensionMismatch("window dm has size $(size(window.dm)), expected $((Nx, Ny, nlevel))"))
    end
    return nothing
end

function _transport_validate_optional_faceindexed_deltas(window, ncell::Int, nface_h::Int, nlevel::Int)
    if haskey(window, :dhflux)
        size(window.dhflux) == (nface_h, nlevel) ||
            throw(DimensionMismatch("window dhflux has size $(size(window.dhflux)), expected $((nface_h, nlevel))"))
    end
    if haskey(window, :dcm)
        size(window.dcm) == (ncell, nlevel + 1) ||
            throw(DimensionMismatch("window dcm has size $(size(window.dcm)), expected $((ncell, nlevel + 1))"))
    end
    if haskey(window, :dm)
        size(window.dm) == (ncell, nlevel) ||
            throw(DimensionMismatch("window dm has size $(size(window.dm)), expected $((ncell, nlevel))"))
    end
    return nothing
end

function _transport_validate_structured_window(window,
                                               Nx::Int, Ny::Int, nlevel::Int,
                                               basis_sym::Symbol)
    m = _transport_window_mass(window)
    am = _transport_window_am(window)
    bm = _transport_window_bm(window)
    cm = _transport_window_cm(window)
    ps = _transport_window_ps(window)

    size(m) == (Nx, Ny, nlevel) ||
        throw(DimensionMismatch("window m has size $(size(m)), expected ($(Nx), $(Ny), $(nlevel))"))
    size(am) == (Nx + 1, Ny, nlevel) ||
        throw(DimensionMismatch("window am has size $(size(am)), expected ($(Nx + 1), $(Ny), $(nlevel))"))
    size(bm) == (Nx, Ny + 1, nlevel) ||
        throw(DimensionMismatch("window bm has size $(size(bm)), expected ($(Nx), $(Ny + 1), $(nlevel))"))
    size(cm) == (Nx, Ny, nlevel + 1) ||
        throw(DimensionMismatch("window cm has size $(size(cm)), expected ($(Nx), $(Ny), $(nlevel + 1))"))
    size(ps) == (Nx, Ny) ||
        throw(DimensionMismatch("window ps has size $(size(ps)), expected ($(Nx), $(Ny))"))

    _transport_validate_optional_qv(window, (Nx, Ny, nlevel))
    _transport_validate_optional_structured_deltas(window, Nx, Ny, nlevel)
    _transport_validate_basis(window, basis_sym)
    return nothing
end

function _transport_validate_reduced_window(window,
                                            ncell::Int, nface_h::Int, nlevel::Int,
                                            basis_sym::Symbol)
    m = _transport_window_mass(window)
    hflux = _transport_window_hflux(window)
    cm = _transport_window_cm(window)
    ps = _transport_window_ps(window)

    size(m) == (ncell, nlevel) ||
        throw(DimensionMismatch("window m has size $(size(m)), expected ($(ncell), $(nlevel))"))
    size(hflux) == (nface_h, nlevel) ||
        throw(DimensionMismatch("window hflux has size $(size(hflux)), expected ($(nface_h), $(nlevel))"))
    size(cm) == (ncell, nlevel + 1) ||
        throw(DimensionMismatch("window cm has size $(size(cm)), expected ($(ncell), $(nlevel + 1))"))
    size(ps) == (ncell,) ||
        throw(DimensionMismatch("window ps has size $(size(ps)), expected ($(ncell),)"))

    _transport_validate_optional_qv(window, (ncell, nlevel))
    _transport_validate_optional_faceindexed_deltas(window, ncell, nface_h, nlevel)
    _transport_validate_basis(window, basis_sym)
    return nothing
end

function _transport_common_header(grid_type::String,
                                  horizontal_topology::String,
                                  ncell::Int,
                                  nface_h::Int,
                                  nlevel::Int,
                                  nwindow::Int,
                                  vc,
                                  payload_sections::Vector{Symbol},
                                  elems_per_window::Int;
                                  FT::Type{<:AbstractFloat},
                                  header_bytes::Int,
                                  dt_met_seconds::Real,
                                  half_dt_seconds::Real,
                                  steps_per_window::Integer,
                                  mass_basis::Symbol,
                                  source_flux_sampling::Symbol,
                                  air_mass_sampling::Symbol,
                                  flux_sampling::Symbol,
                                  flux_kind::Symbol,
                                  humidity_sampling::Symbol,
                                  delta_semantics::Symbol)
    n_qv = (:qv in payload_sections) ? ncell * nlevel : 0
    n_qv_start = (:qv_start in payload_sections) ? ncell * nlevel : 0
    n_qv_end = (:qv_end in payload_sections) ? ncell * nlevel : 0
    humidity_sampling = humidity_sampling === :auto ? _transport_default_humidity_sampling(payload_sections) : _transport_normalize_symbol(humidity_sampling)
    delta_semantics = delta_semantics === :auto ? _transport_default_delta_semantics(payload_sections) : _transport_normalize_symbol(delta_semantics)

    return Dict{String, Any}(
        "magic" => "MFLX",
        "format_version" => 1,
        "header_bytes" => header_bytes,
        "float_type" => string(FT),
        "float_bytes" => sizeof(FT),
        "grid_type" => grid_type,
        "horizontal_topology" => horizontal_topology,
        "ncell" => ncell,
        "nface_h" => nface_h,
        "nlevel" => nlevel,
        "nwindow" => nwindow,
        "vertical_coordinate_type" => "hybrid_sigma_pressure",
        "A_ifc" => Float64.(vc.A),
        "B_ifc" => Float64.(vc.B),
        "dt_met_seconds" => Float64(dt_met_seconds),
        "half_dt_seconds" => Float64(half_dt_seconds),
        "steps_per_window" => Int(steps_per_window),
        "source_flux_sampling" => String(_transport_validate_source_flux_sampling(source_flux_sampling)),
        "air_mass_sampling" => String(_transport_normalize_symbol(air_mass_sampling)),
        "flux_sampling" => String(_transport_normalize_symbol(flux_sampling)),
        "flux_kind" => String(_transport_normalize_symbol(flux_kind)),
        "humidity_sampling" => String(humidity_sampling),
        "delta_semantics" => String(delta_semantics),
        "mass_basis" => String(mass_basis),
        "payload_sections" => String.(payload_sections),
        "include_qv" => :qv in payload_sections,
        "include_qv_endpoints" => (:qv_start in payload_sections) || (:qv_end in payload_sections),
        "include_flux_delta" => any(section in (:dam, :dbm, :dcm, :dm, :dhflux) for section in payload_sections),
        "n_qv" => n_qv,
        "n_qv_start" => n_qv_start,
        "n_qv_end" => n_qv_end,
        "n_geometry_elems" => 0,
        "elems_per_window" => elems_per_window,
    )
end

@inline function _transport_copy_flat!(dest::Vector{FT}, offset::Int, src) where FT
    src_lin = vec(src)
    n = length(src_lin)
    @inbounds for idx in 1:n
        dest[offset + idx] = convert(FT, src_lin[idx])
    end
    return offset + n
end

function _transport_pack_window!(dest::Vector{FT},
                                 window_offset::Int,
                                 window,
                                 payload_sections::Vector{Symbol}) where FT
    offset = window_offset
    for section in payload_sections
        offset = _transport_copy_flat!(dest, offset, _transport_window_field(window, section))
    end
    return nothing
end

function _transport_pack_payload(windows::AbstractVector,
                                 payload_sections::Vector{Symbol},
                                 elems_per_window::Int,
                                 ::Type{FT};
                                 threaded::Bool = Threads.nthreads() > 1) where FT
    nwindows = length(windows)
    payload = Vector{FT}(undef, nwindows * elems_per_window)

    if threaded && nwindows > 1
        Threads.@threads for win in eachindex(windows)
            window_offset = (win - 1) * elems_per_window
            _transport_pack_window!(payload, window_offset, windows[win], payload_sections)
        end
    else
        @inbounds for win in eachindex(windows)
            window_offset = (win - 1) * elems_per_window
            _transport_pack_window!(payload, window_offset, windows[win], payload_sections)
        end
    end

    return payload
end

function _write_transport_payload!(io::IO,
                                   windows::AbstractVector,
                                   payload_sections::Vector{Symbol},
                                   elems_per_window::Int,
                                   ::Type{FT};
                                   threaded::Bool = Threads.nthreads() > 1) where FT
    payload = _transport_pack_payload(windows, payload_sections, elems_per_window, FT; threaded=threaded)
    write(io, payload)
    return nothing
end

function write_transport_binary(path::AbstractString,
                                grid::AtmosGrid{<:LatLonMesh},
                                windows::AbstractVector;
                                FT::Type{<:AbstractFloat} = floattype(grid),
                                header_bytes::Int = 16384,
                                dt_met_seconds::Real = 3600.0,
                                half_dt_seconds::Real = dt_met_seconds / 2,
                                steps_per_window::Integer = 2,
                                source_flux_sampling::Symbol,
                                air_mass_sampling::Symbol = :window_start_endpoint,
                                flux_sampling::Symbol = :window_start_endpoint,
                                flux_kind::Symbol = :substep_mass_amount,
                                humidity_sampling::Symbol = :auto,
                                delta_semantics::Symbol = :auto,
                                mass_basis::Symbol = :dry,
                                extra_header::AbstractDict{<:AbstractString,<:Any} = Dict{String,Any}(),
                                threaded::Bool = Threads.nthreads() > 1)
    isempty(windows) && throw(ArgumentError("write_transport_binary requires at least one window"))

    mesh = grid.horizontal
    vc = grid.vertical
    Nx = nx(mesh)
    Ny = ny(mesh)
    ncell = ncells(mesh)
    nface_h = nfaces(mesh)
    nlevel = nlevels(grid)
    basis_sym = _transport_basis_symbol(mass_basis)
    payload_sections = _transport_payload_sections(grid, first(windows))

    for window in windows
        _transport_payload_sections(grid, window) == payload_sections ||
            throw(ArgumentError("all transport-binary windows must carry the same payload sections"))
        _transport_validate_structured_window(window, Nx, Ny, nlevel, basis_sym)
    end

    elems_per_window = sum(_transport_structured_section_elements(Nx, Ny, ncell, nlevel, section)
                           for section in payload_sections)

    header = _transport_common_header("latlon", "StructuredDirectional",
                                      ncell, nface_h, nlevel, length(windows), vc,
                                      payload_sections, elems_per_window;
                                      FT=FT,
                                      header_bytes=header_bytes,
                                      dt_met_seconds=dt_met_seconds,
                                      half_dt_seconds=half_dt_seconds,
                                      steps_per_window=steps_per_window,
                                      mass_basis=basis_sym,
                                      source_flux_sampling=source_flux_sampling,
                                      air_mass_sampling=air_mass_sampling,
                                      flux_sampling=flux_sampling,
                                      flux_kind=flux_kind,
                                      humidity_sampling=humidity_sampling,
                                      delta_semantics=delta_semantics)
    merge!(header, Dict{String, Any}(
        "Nx" => Nx,
        "Ny" => Ny,
        "lons" => Float64.(mesh.λᶜ),
        "lats" => Float64.(mesh.φᶜ),
        "longitude_interval" => Float64[mesh.λᶠ[1], mesh.λᶠ[end]],
        "latitude_interval" => Float64[mesh.φᶠ[1], mesh.φᶠ[end]],
        "grid_convention" => "south_to_north_periodic_longitude",
        "n_m" => ncell * nlevel,
        "n_am" => (Nx + 1) * Ny * nlevel,
        "n_bm" => Nx * (Ny + 1) * nlevel,
        "n_cm" => ncell * (nlevel + 1),
        "n_ps" => ncell,
        "n_dam" => (:dam in payload_sections) ? _transport_structured_section_elements(Nx, Ny, ncell, nlevel, :dam) : 0,
        "n_dbm" => (:dbm in payload_sections) ? _transport_structured_section_elements(Nx, Ny, ncell, nlevel, :dbm) : 0,
        "n_dcm" => (:dcm in payload_sections) ? _transport_structured_section_elements(Nx, Ny, ncell, nlevel, :dcm) : 0,
        "n_dm" => (:dm in payload_sections) ? _transport_structured_section_elements(Nx, Ny, ncell, nlevel, :dm) : 0,
    ))
    isempty(extra_header) || merge!(header, Dict{String, Any}(extra_header))

    header_json = JSON3.write(header)
    pad = header_bytes - ncodeunits(header_json)
    pad >= 0 || error("transport binary header exceeds header_bytes=$(header_bytes)")

    open(path, "w") do io
        write(io, header_json)
        write(io, zeros(UInt8, pad))
        _write_transport_payload!(io, windows, payload_sections, elems_per_window, FT; threaded=threaded)
    end

    return path
end

function write_transport_binary(path::AbstractString,
                                grid::AtmosGrid{<:ReducedGaussianMesh},
                                windows::AbstractVector;
                                FT::Type{<:AbstractFloat} = floattype(grid),
                                header_bytes::Int = 131072,
                                dt_met_seconds::Real = 3600.0,
                                half_dt_seconds::Real = dt_met_seconds / 2,
                                steps_per_window::Integer = 2,
                                source_flux_sampling::Symbol,
                                air_mass_sampling::Symbol = :window_start_endpoint,
                                flux_sampling::Symbol = :window_start_endpoint,
                                flux_kind::Symbol = :substep_mass_amount,
                                humidity_sampling::Symbol = :auto,
                                delta_semantics::Symbol = :auto,
                                mass_basis::Symbol = :dry,
                                extra_header::AbstractDict{<:AbstractString,<:Any} = Dict{String,Any}(),
                                threaded::Bool = Threads.nthreads() > 1)
    isempty(windows) && throw(ArgumentError("write_transport_binary requires at least one window"))

    mesh = grid.horizontal
    vc = grid.vertical
    ncell = ncells(mesh)
    nface_h = nfaces(mesh)
    nlevel = nlevels(grid)
    basis_sym = _transport_basis_symbol(mass_basis)
    payload_sections = _transport_payload_sections(grid, first(windows))

    for window in windows
        _transport_payload_sections(grid, window) == payload_sections ||
            throw(ArgumentError("all transport-binary windows must carry the same payload sections"))
        _transport_validate_reduced_window(window, ncell, nface_h, nlevel, basis_sym)
    end

    elems_per_window = sum(_transport_faceindexed_section_elements(ncell, nface_h, nlevel, section)
                           for section in payload_sections)

    header = _transport_common_header("reduced_gaussian", "FaceIndexed",
                                      ncell, nface_h, nlevel, length(windows), vc,
                                      payload_sections, elems_per_window;
                                      FT=FT,
                                      header_bytes=header_bytes,
                                      dt_met_seconds=dt_met_seconds,
                                      half_dt_seconds=half_dt_seconds,
                                      steps_per_window=steps_per_window,
                                      mass_basis=basis_sym,
                                      source_flux_sampling=source_flux_sampling,
                                      air_mass_sampling=air_mass_sampling,
                                      flux_sampling=flux_sampling,
                                      flux_kind=flux_kind,
                                      humidity_sampling=humidity_sampling,
                                      delta_semantics=delta_semantics)
    merge!(header, Dict{String, Any}(
        "nlat" => nrings(mesh),
        "latitudes" => Float64.(mesh.latitudes),
        "nlon_per_ring" => mesh.nlon_per_ring,
        "n_dhflux" => (:dhflux in payload_sections) ? _transport_faceindexed_section_elements(ncell, nface_h, nlevel, :dhflux) : 0,
        "n_dcm" => (:dcm in payload_sections) ? _transport_faceindexed_section_elements(ncell, nface_h, nlevel, :dcm) : 0,
        "n_dm" => (:dm in payload_sections) ? _transport_faceindexed_section_elements(ncell, nface_h, nlevel, :dm) : 0,
    ))
    isempty(extra_header) || merge!(header, Dict{String, Any}(extra_header))

    header_json = JSON3.write(header)
    pad = header_bytes - ncodeunits(header_json)
    pad >= 0 || error("transport binary header exceeds header_bytes=$(header_bytes)")

    open(path, "w") do io
        write(io, header_json)
        write(io, zeros(UInt8, pad))
        _write_transport_payload!(io, windows, payload_sections, elems_per_window, FT; threaded=threaded)
    end

    return path
end

# =========================================================================
# Streaming (per-window) binary writer
# =========================================================================

"""
    StreamingTransportBinaryWriter{FT}

Handle for incrementally writing transport-binary windows to disk without
holding all windows in memory.  Created by [`open_streaming_transport_binary`](@ref),
each window is written via [`write_streaming_window!`](@ref), and the file is
finalised by [`close_streaming_transport_binary!`](@ref).

Memory footprint: one `elems_per_window`-length pack buffer (`Vector{FT}`)
plus the open `IOStream`.  All other per-window data is owned by the caller.
"""
mutable struct StreamingTransportBinaryWriter{FT}
    io::IOStream
    path::String
    payload_sections::Vector{Symbol}
    elems_per_window::Int
    header_bytes::Int
    expected_windows::Int
    written_windows::Int
    pack_buffer::Vector{FT}
end

"""
    open_streaming_transport_binary(path, grid::AtmosGrid{<:ReducedGaussianMesh},
                                    nwindow, sample_window; kwargs...)

Open a transport binary file for streaming (per-window) writes on a
reduced-Gaussian grid.

`sample_window` is a NamedTuple with the same keys as the windows that will
be written (e.g. `(m=..., hflux=..., cm=..., ps=...)`).  Its arrays must
have the correct sizes but their *values* are ignored — it is only used to
determine `payload_sections` and to validate dimensions.

Returns a [`StreamingTransportBinaryWriter`](@ref).
"""
function open_streaming_transport_binary(
        path::AbstractString,
        grid::AtmosGrid{<:ReducedGaussianMesh},
        nwindow::Int,
        sample_window;
        FT::Type{<:AbstractFloat} = floattype(grid),
        header_bytes::Int = 131072,
        dt_met_seconds::Real = 3600.0,
        half_dt_seconds::Real = dt_met_seconds / 2,
        steps_per_window::Integer = 2,
        source_flux_sampling::Symbol,
        air_mass_sampling::Symbol = :window_start_endpoint,
        flux_sampling::Symbol = :window_start_endpoint,
        flux_kind::Symbol = :substep_mass_amount,
        humidity_sampling::Symbol = :auto,
        delta_semantics::Symbol = :auto,
        mass_basis::Symbol = :moist,
        extra_header::AbstractDict{<:AbstractString,<:Any} = Dict{String,Any}())

    mesh = grid.horizontal
    vc   = grid.vertical
    ncell   = ncells(mesh)
    nface_h = nfaces(mesh)
    nlevel  = nlevels(grid)
    basis_sym = _transport_basis_symbol(mass_basis)
    payload_sections = _transport_payload_sections(grid, sample_window)

    _transport_validate_reduced_window(sample_window, ncell, nface_h, nlevel, basis_sym)

    elems_per_window = sum(_transport_faceindexed_section_elements(ncell, nface_h, nlevel, s)
                           for s in payload_sections)

    # Build header — identical to the non-streaming write_transport_binary.
    header = _transport_common_header("reduced_gaussian", "FaceIndexed",
                                      ncell, nface_h, nlevel, nwindow, vc,
                                      payload_sections, elems_per_window;
                                      FT=FT,
                                      header_bytes=header_bytes,
                                      dt_met_seconds=dt_met_seconds,
                                      half_dt_seconds=half_dt_seconds,
                                      steps_per_window=steps_per_window,
                                      mass_basis=basis_sym,
                                      source_flux_sampling=source_flux_sampling,
                                      air_mass_sampling=air_mass_sampling,
                                      flux_sampling=flux_sampling,
                                      flux_kind=flux_kind,
                                      humidity_sampling=humidity_sampling,
                                      delta_semantics=delta_semantics)
    merge!(header, Dict{String, Any}(
        "nlat"          => nrings(mesh),
        "latitudes"     => Float64.(mesh.latitudes),
        "nlon_per_ring" => mesh.nlon_per_ring,
        "n_dhflux" => (:dhflux in payload_sections) ?
            _transport_faceindexed_section_elements(ncell, nface_h, nlevel, :dhflux) : 0,
        "n_dcm" => (:dcm in payload_sections) ?
            _transport_faceindexed_section_elements(ncell, nface_h, nlevel, :dcm) : 0,
        "n_dm" => (:dm in payload_sections) ?
            _transport_faceindexed_section_elements(ncell, nface_h, nlevel, :dm) : 0,
    ))
    isempty(extra_header) || merge!(header, Dict{String, Any}(extra_header))

    header_json = JSON3.write(header)
    pad = header_bytes - ncodeunits(header_json)
    pad >= 0 || error("transport binary header exceeds header_bytes=$(header_bytes)")

    io = open(path, "w")
    write(io, header_json)
    write(io, zeros(UInt8, pad))

    pack_buffer = Vector{FT}(undef, elems_per_window)

    return StreamingTransportBinaryWriter{FT}(
        io, String(path), payload_sections, elems_per_window,
        header_bytes, nwindow, 0, pack_buffer)
end

"""
    write_streaming_window!(writer, window)

Pack and write a single window to the streaming transport binary.
Windows must be written in order (1, 2, …, nwindow).
"""
function write_streaming_window!(writer::StreamingTransportBinaryWriter{FT},
                                  window) where FT
    writer.written_windows >= writer.expected_windows &&
        error("Already wrote $(writer.written_windows)/$(writer.expected_windows) windows")
    _transport_pack_window!(writer.pack_buffer, 0, window, writer.payload_sections)
    write(writer.io, writer.pack_buffer)
    writer.written_windows += 1
    return nothing
end

"""
    close_streaming_transport_binary!(writer) -> String

Flush and close the streaming transport binary.  Returns the file path.
Warns if the number of windows written does not match the expected count.
"""
function close_streaming_transport_binary!(writer::StreamingTransportBinaryWriter)
    writer.written_windows == writer.expected_windows ||
        @warn("Streaming binary: expected $(writer.expected_windows) windows, " *
              "wrote $(writer.written_windows)")
    close(writer.io)
    return writer.path
end

# =========================================================================
# CS streaming writer
# =========================================================================

"""
    _cs_section_elements(Nc, npanel, nlevel, section) -> Int

Return the number of float elements for a given section in a CS binary.
Panels are stored sequentially within each section.
"""
function _cs_section_elements(Nc::Int, npanel::Int, nlevel::Int, section::Symbol)
    if section === :m
        return npanel * Nc * Nc * nlevel
    elseif section === :am
        return npanel * (Nc + 1) * Nc * nlevel
    elseif section === :bm
        return npanel * Nc * (Nc + 1) * nlevel
    elseif section === :cm
        return npanel * Nc * Nc * (nlevel + 1)
    elseif section === :ps
        return npanel * Nc * Nc
    elseif section === :cmfmc
        return npanel * Nc * Nc * (nlevel + 1)
    elseif section === :dtrain
        return npanel * Nc * Nc * nlevel
    else
        error("Unsupported CS section: $section")
    end
end

"""
    _pack_cs_window!(dest, offset, window, payload_sections, Nc, npanel)

Pack a CS window (with NTuple-of-panels fields) into a flat buffer.
Each section's panels are stored sequentially: [P1][P2]...[P6].
"""
function _pack_cs_window!(dest::Vector{FT}, offset::Int,
                           window, payload_sections::Vector{Symbol},
                           Nc::Int, npanel::Int) where FT
    o = offset
    for section in payload_sections
        panels = getfield(window, section)
        for p in 1:npanel
            panel_data = panels[p]
            n = length(panel_data)
            @inbounds for idx in 1:n
                dest[o + idx] = convert(FT, panel_data[idx])
            end
            o += n
        end
    end
    return nothing
end

"""
    open_streaming_cs_transport_binary(path, Nc, npanel, nlevel, nwindow, vc;
                                       kwargs...) -> StreamingTransportBinaryWriter

Open a CS transport binary for streaming per-window writes.

`vc` is a `HybridSigmaPressure` vertical coordinate. The CS binary uses
per-panel structured arrays with `StructuredDirectional` topology.
"""
function open_streaming_cs_transport_binary(
        path::AbstractString,
        Nc::Int,
        npanel::Int,
        nlevel::Int,
        nwindow::Int,
        vc;
        FT::Type{<:AbstractFloat} = Float64,
        header_bytes::Int = 131072,
        dt_met_seconds::Real = 3600.0,
        half_dt_seconds::Real = dt_met_seconds / 2,
        steps_per_window::Integer = 4,
        source_flux_sampling::Symbol = :window_start_endpoint,
        air_mass_sampling::Symbol = :window_start_endpoint,
        flux_sampling::Symbol = :window_constant,
        flux_kind::Symbol = :substep_mass_amount,
        mass_basis::Symbol = :moist,
        include_cmfmc::Bool = false,
        include_dtrain::Bool = false,
        extra_header::AbstractDict{<:AbstractString,<:Any} = Dict{String,Any}())
    include_dtrain && !include_cmfmc &&
        throw(ArgumentError("CS transport binaries cannot include dtrain without cmfmc"))

    ncell = npanel * Nc * Nc
    nface_h = npanel * 2 * Nc * (Nc + 1)
    payload_sections = Symbol[:m, :am, :bm, :cm, :ps]
    include_cmfmc && push!(payload_sections, :cmfmc)
    include_dtrain && push!(payload_sections, :dtrain)

    elems_per_window = sum(_cs_section_elements(Nc, npanel, nlevel, s)
                           for s in payload_sections)

    header = _transport_common_header("cubed_sphere", "StructuredDirectional",
                                      ncell, nface_h, nlevel, nwindow, vc,
                                      payload_sections, elems_per_window;
                                      FT=FT,
                                      header_bytes=header_bytes,
                                      dt_met_seconds=dt_met_seconds,
                                      half_dt_seconds=half_dt_seconds,
                                      steps_per_window=steps_per_window,
                                      mass_basis=mass_basis,
                                      source_flux_sampling=source_flux_sampling,
                                      air_mass_sampling=air_mass_sampling,
                                      flux_sampling=flux_sampling,
                                      flux_kind=flux_kind,
                                      humidity_sampling=:none,
                                      delta_semantics=:none)

    merge!(header, Dict{String, Any}(
        "Nc" => Nc,
        "npanel" => npanel,
        "panel_convention" => "gnomonic",
        "Hp" => 0,
        "poisson_balance_method" => "global_cg_graph_laplacian",
        "poisson_balance_target_scale" => 1.0 / (2 * steps_per_window),
        "poisson_balance_target_semantics" => "forward_window_mass_difference / (2 * steps_per_window)",
        "include_cmfmc" => include_cmfmc,
        "include_dtrain" => include_dtrain,
        "n_cmfmc" => include_cmfmc ? _cs_section_elements(Nc, npanel, nlevel, :cmfmc) : 0,
        "n_dtrain" => include_dtrain ? _cs_section_elements(Nc, npanel, nlevel, :dtrain) : 0,
    ))
    isempty(extra_header) || merge!(header, Dict{String, Any}(extra_header))

    header_json = JSON3.write(header)
    pad = header_bytes - ncodeunits(header_json)
    pad >= 0 || error("transport binary header exceeds header_bytes=$(header_bytes)")

    io = open(path, "w")
    write(io, header_json)
    write(io, zeros(UInt8, pad))

    pack_buffer = Vector{FT}(undef, elems_per_window)

    return StreamingTransportBinaryWriter{FT}(
        io, String(path), payload_sections, elems_per_window,
        header_bytes, nwindow, 0, pack_buffer)
end

"""
    write_streaming_cs_window!(writer, window, Nc, npanel)

Pack and write a single CS window to the streaming transport binary.
`window` is a NamedTuple with NTuple-of-panels fields `:m`, `:am`, `:bm`, `:cm`, `:ps`.
"""
function write_streaming_cs_window!(writer::StreamingTransportBinaryWriter{FT},
                                     window, Nc::Int, npanel::Int) where FT
    writer.written_windows >= writer.expected_windows &&
        error("Already wrote $(writer.written_windows)/$(writer.expected_windows) windows")
    _pack_cs_window!(writer.pack_buffer, 0, window, writer.payload_sections, Nc, npanel)
    write(writer.io, writer.pack_buffer)
    writer.written_windows += 1
    return nothing
end

function _transport_interval_from_centers(centers::Vector{Float64}, fallback_Δ::Float64)
    isempty(centers) && error("Cannot reconstruct interval from empty center array")
    Δ = length(centers) > 1 ? centers[2] - centers[1] : fallback_Δ
    return (centers[1] - Δ / 2, centers[end] + Δ / 2)
end

function load_grid(reader::TransportBinaryReader;
                   FT::Type{<:AbstractFloat} = Float64,
                   arch = CPU())
    h = reader.header
    vc = HybridSigmaPressure(FT.(h.A_ifc), FT.(h.B_ifc))

    if _transport_is_structured(h)
        longitude = length(h.longitude_interval_f64) == 2 ?
            (FT(h.longitude_interval_f64[1]), FT(h.longitude_interval_f64[2])) :
            let interval = _transport_interval_from_centers(h.lons_f64, 360.0 / h.Nx)
                (FT(interval[1]), FT(interval[2]))
            end
        latitude = length(h.latitude_interval_f64) == 2 ?
            (FT(h.latitude_interval_f64[1]), FT(h.latitude_interval_f64[2])) :
            let interval = _transport_interval_from_centers(h.lats_f64, 180.0 / h.Ny)
                (FT(interval[1]), FT(interval[2]))
            end
        mesh = LatLonMesh(; FT=FT, size=(h.Nx, h.Ny), longitude=longitude, latitude=latitude)
        return AtmosGrid(mesh, vc, arch; FT=FT)
    elseif _transport_is_faceindexed(h)
        mesh = ReducedGaussianMesh(h.ring_latitudes_f64, h.nlon_per_ring; FT=FT)
        return AtmosGrid(mesh, vc, arch; FT=FT)
    else
        throw(ArgumentError("Unsupported transport binary grid/topology combination: $(h.grid_type) / $(h.horizontal_topology)"))
    end
end

_transport_allocate_mass(reader::TransportBinaryReader{FT}) where FT =
    _transport_is_structured(reader.header) ?
        Array{FT}(undef, reader.header.Nx, reader.header.Ny, reader.header.nlevel) :
        Array{FT}(undef, reader.header.ncell, reader.header.nlevel)

_transport_allocate_ps(reader::TransportBinaryReader{FT}) where FT =
    _transport_is_structured(reader.header) ?
        Array{FT}(undef, reader.header.Nx, reader.header.Ny) :
        Array{FT}(undef, reader.header.ncell)

_transport_allocate_cm(reader::TransportBinaryReader{FT}) where FT =
    _transport_is_structured(reader.header) ?
        Array{FT}(undef, reader.header.Nx, reader.header.Ny, reader.header.nlevel + 1) :
        Array{FT}(undef, reader.header.ncell, reader.header.nlevel + 1)

_transport_allocate_am(reader::TransportBinaryReader{FT}) where FT =
    Array{FT}(undef, reader.header.Nx + 1, reader.header.Ny, reader.header.nlevel)

_transport_allocate_bm(reader::TransportBinaryReader{FT}) where FT =
    Array{FT}(undef, reader.header.Nx, reader.header.Ny + 1, reader.header.nlevel)

_transport_allocate_hflux(reader::TransportBinaryReader{FT}) where FT =
    Array{FT}(undef, reader.header.nface_h, reader.header.nlevel)

_transport_allocate_qv(reader::TransportBinaryReader{FT}) where FT =
    _transport_is_structured(reader.header) ?
        Array{FT}(undef, reader.header.Nx, reader.header.Ny, reader.header.nlevel) :
        Array{FT}(undef, reader.header.ncell, reader.header.nlevel)

_transport_allocate_dam(reader::TransportBinaryReader{FT}) where FT =
    Array{FT}(undef, reader.header.Nx + 1, reader.header.Ny, reader.header.nlevel)

_transport_allocate_dbm(reader::TransportBinaryReader{FT}) where FT =
    Array{FT}(undef, reader.header.Nx, reader.header.Ny + 1, reader.header.nlevel)

_transport_allocate_dhflux(reader::TransportBinaryReader{FT}) where FT =
    Array{FT}(undef, reader.header.nface_h, reader.header.nlevel)

_transport_allocate_dm(reader::TransportBinaryReader{FT}) where FT =
    _transport_is_structured(reader.header) ?
        Array{FT}(undef, reader.header.Nx, reader.header.Ny, reader.header.nlevel) :
        Array{FT}(undef, reader.header.ncell, reader.header.nlevel)

_transport_allocate_dcm(reader::TransportBinaryReader{FT}) where FT =
    _transport_is_structured(reader.header) ?
        Array{FT}(undef, reader.header.Nx, reader.header.Ny, reader.header.nlevel + 1) :
        Array{FT}(undef, reader.header.ncell, reader.header.nlevel + 1)

# TM5 convection fields — all layer-center, shape matches `m`
# (plan 23 Commit 3).
_transport_allocate_tm5_field(reader::TransportBinaryReader{FT}) where FT =
    _transport_is_structured(reader.header) ?
        Array{FT}(undef, reader.header.Nx, reader.header.Ny, reader.header.nlevel) :
        Array{FT}(undef, reader.header.ncell, reader.header.nlevel)

function _transport_make_fluxes(::Val{:dry}, am, bm, cm)
    return StructuredFaceFluxState{DryMassFluxBasis}(am, bm, cm)
end

function _transport_make_fluxes(::Val{:moist}, am, bm, cm)
    return StructuredFaceFluxState{MoistMassFluxBasis}(am, bm, cm)
end

function _transport_make_fluxes(::Val{:dry}, hflux, cm)
    return FaceIndexedFluxState{DryMassFluxBasis}(hflux, cm)
end

function _transport_make_fluxes(::Val{:moist}, hflux, cm)
    return FaceIndexedFluxState{MoistMassFluxBasis}(hflux, cm)
end

function load_window!(reader::TransportBinaryReader{FT}, win::Int;
                      m = nothing,
                      ps = nothing,
                      hflux = nothing,
                      am = nothing,
                      bm = nothing,
                      cm = nothing) where FT
    h = reader.header
    m = isnothing(m) ? _transport_allocate_mass(reader) : m
    ps = isnothing(ps) ? _transport_allocate_ps(reader) : ps
    cm = isnothing(cm) ? _transport_allocate_cm(reader) : cm

    if _transport_is_structured(h)
        am = isnothing(am) ? _transport_allocate_am(reader) : am
        bm = isnothing(bm) ? _transport_allocate_bm(reader) : bm
        o = _transport_window_offset(reader, win)
        saw_m = saw_am = saw_bm = saw_cm = saw_ps = false

        for section in h.payload_sections
            n = _transport_section_elements(h, section)
            if section === :m
                copyto!(m, 1, reader.data, o + 1, n)
                saw_m = true
            elseif section === :am
                copyto!(am, 1, reader.data, o + 1, n)
                saw_am = true
            elseif section === :bm
                copyto!(bm, 1, reader.data, o + 1, n)
                saw_bm = true
            elseif section === :cm
                copyto!(cm, 1, reader.data, o + 1, n)
                saw_cm = true
            elseif section === :ps
                copyto!(ps, 1, reader.data, o + 1, n)
                saw_ps = true
            end
            o += n
        end

        saw_m || error("Transport binary payload is missing required section `m`")
        saw_am || error("Transport binary payload is missing required section `am`")
        saw_bm || error("Transport binary payload is missing required section `bm`")
        saw_cm || error("Transport binary payload is missing required section `cm`")
        saw_ps || error("Transport binary payload is missing required section `ps`")

        fluxes = _transport_make_fluxes(Val(h.mass_basis), am, bm, cm)
        return m, ps, fluxes
    elseif _transport_is_faceindexed(h)
        hflux = isnothing(hflux) ? _transport_allocate_hflux(reader) : hflux
        o = _transport_window_offset(reader, win)
        saw_m = saw_hflux = saw_cm = saw_ps = false

        for section in h.payload_sections
            n = _transport_section_elements(h, section)
            if section === :m
                copyto!(m, 1, reader.data, o + 1, n)
                saw_m = true
            elseif section === :hflux
                copyto!(hflux, 1, reader.data, o + 1, n)
                saw_hflux = true
            elseif section === :cm
                copyto!(cm, 1, reader.data, o + 1, n)
                saw_cm = true
            elseif section === :ps
                copyto!(ps, 1, reader.data, o + 1, n)
                saw_ps = true
            end
            o += n
        end

        saw_m || error("Transport binary payload is missing required section `m`")
        saw_hflux || error("Transport binary payload is missing required section `hflux`")
        saw_cm || error("Transport binary payload is missing required section `cm`")
        saw_ps || error("Transport binary payload is missing required section `ps`")

        fluxes = _transport_make_fluxes(Val(h.mass_basis), hflux, cm)
        return m, ps, fluxes
    else
        throw(ArgumentError("Unsupported transport binary grid/topology combination: $(h.grid_type) / $(h.horizontal_topology)"))
    end
end

function load_flux_delta_window!(reader::TransportBinaryReader{FT}, win::Int;
                                 dam = nothing,
                                 dbm = nothing,
                                 dhflux = nothing,
                                 dcm = nothing,
                                 dm = nothing) where FT
    has_flux_delta(reader) || return nothing
    h = reader.header

    dam = isnothing(dam) && _transport_is_structured(h) ? _transport_allocate_dam(reader) : dam
    dbm = isnothing(dbm) && _transport_is_structured(h) ? _transport_allocate_dbm(reader) : dbm
    dhflux = isnothing(dhflux) && _transport_is_faceindexed(h) ? _transport_allocate_dhflux(reader) : dhflux
    dcm = isnothing(dcm) ? _transport_allocate_dcm(reader) : dcm
    dm = isnothing(dm) ? _transport_allocate_dm(reader) : dm

    o = _transport_window_offset(reader, win)
    found_any = false
    found_dam = found_dbm = found_dhflux = found_dcm = found_dm = false
    for section in h.payload_sections
        n = _transport_section_elements(h, section)
        if section === :dam
            copyto!(dam, 1, reader.data, o + 1, n)
            found_dam = true
            found_any = true
        elseif section === :dbm
            copyto!(dbm, 1, reader.data, o + 1, n)
            found_dbm = true
            found_any = true
        elseif section === :dhflux
            copyto!(dhflux, 1, reader.data, o + 1, n)
            found_dhflux = true
            found_any = true
        elseif section === :dcm
            copyto!(dcm, 1, reader.data, o + 1, n)
            found_dcm = true
            found_any = true
        elseif section === :dm
            copyto!(dm, 1, reader.data, o + 1, n)
            found_dm = true
            found_any = true
        end
        o += n
    end

    found_any || return nothing

    result = NamedTuple()
    if found_dam
        result = merge(result, (; dam))
    end
    if found_dbm
        result = merge(result, (; dbm))
    end
    if found_dhflux
        result = merge(result, (; dhflux))
    end
    if found_dcm
        result = merge(result, (; dcm))
    end
    if found_dm
        result = merge(result, (; dm))
    end
    return result
end

"""
    load_tm5_convection_window!(reader, win; entu=..., detu=..., entd=..., detd=...) -> NamedTuple | nothing

Load the four TM5 convection layer-center fields for window `win`.
Returns `(; entu, detu, entd, detd)` when the binary carries all
four sections, or `nothing` when no TM5 data is present. Allocates
only if the caller doesn't provide pre-allocated buffers.

All fields share the same shape as `m`: `(Nx, Ny, Nz)` for
structured or `(ncells, Nz)` for face-indexed binaries. Orientation
is as written by the preprocessor (AtmosTransport: k=1=TOA,
k=Nz=surface); no runtime reorientation happens here — the kernel
(plan 23 Commit 4) reads them directly.

Invariant: if ANY of the four sections is present in the header,
ALL four must be present. This mirrors the
`ConvectionForcing.tm5_fields` NamedTuple contract — partial
payload is not a valid convection forcing.
"""
function load_tm5_convection_window!(reader::TransportBinaryReader{FT}, win::Int;
                                      entu = nothing,
                                      detu = nothing,
                                      entd = nothing,
                                      detd = nothing) where FT
    has_tm5_convection(reader) || return nothing
    h = reader.header

    entu = isnothing(entu) ? _transport_allocate_tm5_field(reader) : entu
    detu = isnothing(detu) ? _transport_allocate_tm5_field(reader) : detu
    entd = isnothing(entd) ? _transport_allocate_tm5_field(reader) : entd
    detd = isnothing(detd) ? _transport_allocate_tm5_field(reader) : detd

    o = _transport_window_offset(reader, win)
    for section in h.payload_sections
        n = _transport_section_elements(h, section)
        if section === :entu
            copyto!(entu, 1, reader.data, o + 1, n)
        elseif section === :detu
            copyto!(detu, 1, reader.data, o + 1, n)
        elseif section === :entd
            copyto!(entd, 1, reader.data, o + 1, n)
        elseif section === :detd
            copyto!(detd, 1, reader.data, o + 1, n)
        end
        o += n
    end
    return (; entu, detu, entd, detd)
end

function load_qv_window!(reader::TransportBinaryReader{FT}, win::Int;
                         qv = nothing) where FT
    h = reader.header
    h.include_qv || return nothing
    qv = isnothing(qv) ? _transport_allocate_qv(reader) : qv

    o = _transport_window_offset(reader, win)
    for section in h.payload_sections
        n = _transport_section_elements(h, section)
        if section === :qv
            copyto!(qv, 1, reader.data, o + 1, n)
            return qv
        end
        o += n
    end

    return nothing
end

function load_qv_pair_window!(reader::TransportBinaryReader{FT}, win::Int;
                              qv_start = nothing,
                              qv_end = nothing) where FT
    h = reader.header
    h.include_qv_endpoints || return nothing
    qv_start = isnothing(qv_start) ? _transport_allocate_qv(reader) : qv_start
    qv_end = isnothing(qv_end) ? _transport_allocate_qv(reader) : qv_end

    o = _transport_window_offset(reader, win)
    found_start = false
    found_end = false
    for section in h.payload_sections
        n = _transport_section_elements(h, section)
        if section === :qv_start
            copyto!(qv_start, 1, reader.data, o + 1, n)
            found_start = true
        elseif section === :qv_end
            copyto!(qv_end, 1, reader.data, o + 1, n)
            found_end = true
        end
        o += n
    end

    return found_start && found_end ? (; qv_start, qv_end) : nothing
end

export TransportBinaryHeader, TransportBinaryReader
export grid_type, horizontal_topology, load_grid, load_qv_window!, load_qv_pair_window!, load_flux_delta_window!
export has_qv, has_qv_endpoints, has_flux_delta, write_transport_binary
export source_flux_sampling, air_mass_sampling, flux_sampling, flux_kind, humidity_sampling, delta_semantics
