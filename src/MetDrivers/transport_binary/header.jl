# Header schema, parsing, and on-disk metadata for the transport binary format.
# (TransportBinaryHeader, schedule/sampling parsers, header summary/show, _parse_transport_header, _transport_common_header.)
#
# Part of the TransportBinary format implementation; included from
# `TransportBinary.jl` into the `MetDrivers` module (shared namespace,
# shared `using`s). Split out of the former 2658-line monolith — pure code
# move, no behavior change.

const _PBL_SURFACE_PAYLOAD_SECTIONS = (:pblh, :ustar, :pbl_hflux, :t2m)
const _PBL_SURFACE_FIELD_NAMES = (:pblh, :ustar, :hflux, :t2m)
const _GCHP_VDIFF_PAYLOAD_SECTIONS = (:vdiff_u, :vdiff_v, :vdiff_t, :vdiff_qv)
const _GCHP_VDIFF_FIELD_NAMES = (:u, :v, :t, :qv)
const TRANSPORT_BINARY_FORMAT_VERSION = 4

@inline _is_pbl_surface_payload_section(section::Symbol) =
    section in _PBL_SURFACE_PAYLOAD_SECTIONS
@inline _pbl_surface_field_name(section::Symbol) =
    section === :pbl_hflux ? :hflux : section
@inline _is_gchp_vdiff_payload_section(section::Symbol) =
    section in _GCHP_VDIFF_PAYLOAD_SECTIONS
@inline function _gchp_vdiff_field_name(section::Symbol)
    section === :vdiff_u  && return :u
    section === :vdiff_v  && return :v
    section === :vdiff_t  && return :t
    section === :vdiff_qv && return :qv
    throw(ArgumentError("not a GCHP VDIFF payload section: $(section)"))
end

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
- `flux_kind`: `:substep_mass_amount` or `:full_window_mass_amount` — stored
  flux is either the mass [kg] per scheduled transport substep, or the full
  met-window mass amount that the runtime divides by its schedule.
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
    steps_per_window     :: Int         # scalar compatibility summary (constant value or max schedule)
    steps_per_window_by_window :: Vector{Int}
    source_flux_sampling :: Symbol
    air_mass_sampling    :: Symbol
    flux_sampling        :: Symbol      # :window_constant
    flux_kind            :: Symbol      # :substep_mass_amount or :full_window_mass_amount
    humidity_sampling    :: Symbol
    delta_semantics      :: Symbol
    poisson_balance_target_scale :: Float64
    poisson_balance_target_semantics :: String
    poisson_balance_target_scale_by_window :: Vector{Float64}
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

function _parse_steps_per_window_schedule(hdr, nwindow::Integer,
                                          steps_per_window::Integer)
    haskey(hdr, :steps_per_window_by_window) ||
        throw(ArgumentError("format_version=$(TRANSPORT_BINARY_FORMAT_VERSION) requires " *
                            "`steps_per_window_by_window`; regenerate this binary"))
    schedule = Int.(collect(hdr.steps_per_window_by_window))
    length(schedule) == Int(nwindow) ||
        throw(ArgumentError("steps_per_window_by_window length $(length(schedule)) " *
                            "does not match nwindow=$(nwindow)"))
    all(>=(1), schedule) ||
        throw(ArgumentError("steps_per_window_by_window must contain only positive integers; got $(schedule)"))
    Int(steps_per_window) == maximum(schedule) ||
        throw(ArgumentError("steps_per_window must be maximum(steps_per_window_by_window); " *
                            "got steps_per_window=$(steps_per_window), schedule=$(schedule)"))
    return schedule
end

function _parse_poisson_scale_schedule(hdr, nwindow::Integer,
                                       scalar_scale::Real)
    haskey(hdr, :poisson_balance_target_scale_by_window) ||
        throw(ArgumentError("format_version=$(TRANSPORT_BINARY_FORMAT_VERSION) requires " *
                            "`poisson_balance_target_scale_by_window`; regenerate this binary"))
    scales = Float64.(collect(hdr.poisson_balance_target_scale_by_window))
    length(scales) == Int(nwindow) ||
        throw(ArgumentError("poisson_balance_target_scale_by_window length $(length(scales)) " *
                            "does not match nwindow=$(nwindow)"))
    all(s -> isfinite(s) && s > 0, scales) ||
        throw(ArgumentError("poisson_balance_target_scale_by_window must contain only finite positive values"))
    isfinite(Float64(scalar_scale)) && Float64(scalar_scale) > 0 ||
        throw(ArgumentError("poisson_balance_target_scale must be finite and positive"))
    return scales
end

@inline _has_variable_step_schedule(schedule::AbstractVector{<:Integer}) =
    !isempty(schedule) && any(!=(first(schedule)), schedule)

function _steps_per_window_summary(steps::Integer,
                                   schedule::AbstractVector{<:Integer})
    if _has_variable_step_schedule(schedule)
        return string(Int(steps), " max (per-window ",
                      minimum(schedule), ":", maximum(schedule), ")")
    end
    return string(Int(steps))
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
          "├── timing:        dt=", h.dt_met_seconds, " s, steps/window=",
              _steps_per_window_summary(h.steps_per_window, h.steps_per_window_by_window), "\n",
          "├── payload:       ", join(String.(h.payload_sections), ", "), "\n",
          "├── humidity:      ", _transport_qv_summary(h), "\n",
          "├── semantics:     ", _transport_semantics_summary(h), "\n",
          "├── poisson:       ", isnan(h.poisson_balance_target_scale) ? "unspecified" :
                               string("scale=", h.poisson_balance_target_scale, ", ", h.poisson_balance_target_semantics), "\n",
          "└── header bytes:  ", h.header_bytes)
end

_transport_is_structured(h::TransportBinaryHeader) =
    h.grid_type === :latlon && h.horizontal_topology === :structureddirectional
_transport_is_faceindexed(h::TransportBinaryHeader) =
    h.grid_type === :reduced_gaussian && h.horizontal_topology === :faceindexed

function _transport_disk_float_type(sym::Symbol)
    sym === :Float64 ? Float64 : Float32
end

function _transport_parse_on_disk_float_type(hdr)
    haskey(hdr, :float_type) ||
        throw(ArgumentError("transport binary header is missing float_type"))
    ft_str = string(hdr.float_type)
    if ft_str == "Float64"
        return :Float64, 8
    elseif ft_str == "Float32"
        return :Float32, 4
    end
    throw(ArgumentError("unsupported transport binary float_type=$(repr(ft_str)); expected Float32 or Float64"))
end

function _transport_parse_mass_basis(hdr)
    haskey(hdr, :mass_basis) ||
        throw(ArgumentError("transport binary header is missing mass_basis; regenerate the binary"))
    basis_str = lowercase(string(hdr.mass_basis))
    basis_str in ("dry", "moist") || throw(ArgumentError(
        "unsupported transport binary mass_basis=$(repr(basis_str)); expected dry or moist"))
    return Symbol(basis_str)
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
# one. Silent defaults are how an LL+TM5 binary can land on disk without
# declaring `flux_sampling=:window_constant`: the runtime parser would
# default to `:window_start_endpoint` and run the pre-memo-37 bug class.
# See docs/37_WINDOW_CONSTANT_FLUX_INTERPRETATION_BUG.
# ===========================================================================

const _TRANSPORT_ALLOWED_AIR_MASS_SAMPLINGS = (:window_start_endpoint,)
const _TRANSPORT_ALLOWED_FLUX_SAMPLINGS     = (:window_start_endpoint, :window_constant, :window_mean)
const _TRANSPORT_ALLOWED_FLUX_KINDS         = (:substep_mass_amount, :full_window_mass_amount)
const _TRANSPORT_ALLOWED_DELTA_SEMANTICS    = (:forward_window_endpoint_difference, :none)
const _TRANSPORT_ALLOWED_HUMIDITY_SAMPLINGS = (:window_endpoints, :single_field, :none)

_transport_parse_grid_type(hdr) = Symbol(lowercase(string(hdr.grid_type)))
_transport_parse_topology(hdr) = Symbol(lowercase(string(hdr.horizontal_topology)))

function _transport_parse_sections(hdr)
    sections = haskey(hdr, :payload_sections) ? collect(hdr.payload_sections) : ["m", "hflux", "cm", "ps"]
    return Symbol.(lowercase.(String.(sections)))
end

function _transport_header_axis(hdr, n::Int, key::Symbol)
    haskey(hdr, key) && return Float64.(collect(getproperty(hdr, key)))
    n == 0 && return Float64[]
    error("Transport binary header missing $(key)")
end

function _transport_header_interval(hdr, key::Symbol)
    return haskey(hdr, key) ? Float64.(collect(getproperty(hdr, key))) : Float64[]
end

function _parse_transport_header(raw_bytes::Vector{UInt8})
    json_end = something(findfirst(==(0x00), raw_bytes), length(raw_bytes) + 1) - 1
    hdr = JSON3.read(String(raw_bytes[1:json_end]))

    haskey(hdr, :format_version) ||
        error("TransportBinaryReader requires the topology-generic binary family header (`format_version` missing)")

    # No silent defaults for missing contract fields.
    # `validate_transport_contract!` (called by `TransportBinaryReader`
    # before we get here) has already verified the 8 fields are present —
    # before parsing the typed header.
    format_version = Int(hdr.format_version)
    format_version == TRANSPORT_BINARY_FORMAT_VERSION ||
        throw(ArgumentError("Obsolete transport binary format_version=$(format_version); " *
                            "current runtime requires format_version=$(TRANSPORT_BINARY_FORMAT_VERSION). Regenerate."))
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
    include_qv = :qv in payload_sections
    include_qv_endpoints = (:qv_start in payload_sections) && (:qv_end in payload_sections)
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
    steps_per_window = Int(hdr.steps_per_window)
    steps_schedule = _parse_steps_per_window_schedule(hdr, nwindow, steps_per_window)
    poisson_scale_schedule = _parse_poisson_scale_schedule(
        hdr, nwindow, poisson_balance_target_scale)
    n_qv = Int(get(hdr, :n_qv, include_qv ? ncell * nlevel : 0))
    n_qv_start = Int(get(hdr, :n_qv_start, include_qv_endpoints ? ncell * nlevel : 0))
    n_qv_end = Int(get(hdr, :n_qv_end, include_qv_endpoints ? ncell * nlevel : 0))
    n_geometry_elems = Int(get(hdr, :n_geometry_elems, 0))
    elems_per_window = Int(hdr.elems_per_window)

    Nx = haskey(hdr, :Nx) ? Int(hdr.Nx) : 0
    Ny = haskey(hdr, :Ny) ? Int(hdr.Ny) : 0
    lons_f64 = _transport_header_axis(hdr, Nx, :lons)
    lats_f64 = _transport_header_axis(hdr, Ny, :lats)
    longitude_interval_f64 = _transport_header_interval(hdr, :longitude_interval)
    latitude_interval_f64 = _transport_header_interval(hdr, :latitude_interval)

    nlon_per_ring = haskey(hdr, :nlon_per_ring) ? Int.(collect(hdr.nlon_per_ring)) : Int[]
    ring_latitudes = haskey(hdr, :latitudes) ?
                     Float64.(collect(hdr.latitudes)) : Float64[]
    nlat = haskey(hdr, :nlat) ? Int(hdr.nlat) : 0

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
        steps_per_window,
        steps_schedule,
        source_flux_sampling,
        air_mass_sampling,
        flux_sampling,
        flux_kind,
        humidity_sampling,
        delta_semantics,
        poisson_balance_target_scale,
        poisson_balance_target_semantics,
        poisson_scale_schedule,
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

function _transport_basis_symbol(sym::Symbol)
    basis = Symbol(lowercase(String(sym)))
    basis in (:dry, :moist) ||
        throw(ArgumentError("mass_basis must be :dry or :moist; got $(repr(sym))"))
    return basis
end
_transport_basis_symbol(::DryBasis) = :dry
_transport_basis_symbol(::MoistBasis) = :moist

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
    n_surface = (:pblh in payload_sections) ? ncell : 0
    n_tm5 = (:entu in payload_sections) ? ncell * nlevel : 0
    n_vdiff = (:vdiff_u in payload_sections) ? ncell * nlevel : 0
    humidity_sampling = humidity_sampling === :auto ? _transport_default_humidity_sampling(payload_sections) : _transport_normalize_symbol(humidity_sampling)
    delta_semantics = delta_semantics === :auto ? _transport_default_delta_semantics(payload_sections) : _transport_normalize_symbol(delta_semantics)
    steps = Int(steps_per_window)
    steps > 0 || throw(ArgumentError("steps_per_window must be positive"))
    step_schedule = fill(steps, nwindow)
    poisson_scale_schedule = [1.0 / (2 * s) for s in step_schedule]
    contract = TransportBinaryContract(
        source_flux_sampling = source_flux_sampling,
        air_mass_sampling = air_mass_sampling,
        flux_sampling = flux_sampling,
        flux_kind = flux_kind,
        delta_semantics = delta_semantics,
        humidity_sampling = humidity_sampling,
        poisson_balance_target_scale = 1.0 / (2 * steps),
        poisson_balance_target_semantics = "forward_window_mass_difference / (2 * steps_per_window)",
    )

    return Dict{String, Any}(
        "magic" => "MFLX",
        "format_version" => TRANSPORT_BINARY_FORMAT_VERSION,
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
        "steps_per_window" => steps,
        "steps_per_window_by_window" => step_schedule,
        "time_step_schedule" => "constant",
        "source_flux_sampling" => String(contract.source_flux_sampling),
        "air_mass_sampling" => String(contract.air_mass_sampling),
        "flux_sampling" => String(contract.flux_sampling),
        "flux_kind" => String(contract.flux_kind),
        "humidity_sampling" => String(contract.humidity_sampling),
        "delta_semantics" => String(contract.delta_semantics),
        "poisson_balance_target_scale" => contract.poisson_balance_target_scale,
        "poisson_balance_target_semantics" => contract.poisson_balance_target_semantics,
        "poisson_balance_target_scale_by_window" => poisson_scale_schedule,
        "mass_basis" => String(mass_basis),
        "payload_sections" => String.(payload_sections),
        "include_qv" => :qv in payload_sections,
        "include_qv_endpoints" => (:qv_start in payload_sections) || (:qv_end in payload_sections),
        "include_flux_delta" => any(section in (:dam, :dbm, :dcm, :dm, :dhflux) for section in payload_sections),
        "include_surface" => n_surface > 0,
        "surface_payload" => n_surface > 0 ? "pbl_raw_v2" : "none",
        "include_tm5conv" => n_tm5 > 0,
        "include_gchp_vdiff" => n_vdiff > 0,
        "gchp_vdiff_payload" => n_vdiff > 0 ? "u_v_t_qv_layer_center_v1" : "none",
        "n_qv" => n_qv,
        "n_qv_start" => n_qv_start,
        "n_qv_end" => n_qv_end,
        "n_pblh" => n_surface,
        "n_ustar" => n_surface,
        "n_pbl_hflux" => n_surface,
        "n_t2m" => n_surface,
        "n_entu" => n_tm5,
        "n_detu" => n_tm5,
        "n_entd" => n_tm5,
        "n_detd" => n_tm5,
        "n_vdiff_u" => n_vdiff,
        "n_vdiff_v" => n_vdiff,
        "n_vdiff_t" => n_vdiff,
        "n_vdiff_qv" => n_vdiff,
        "n_geometry_elems" => 0,
        "elems_per_window" => elems_per_window,
    )
end
