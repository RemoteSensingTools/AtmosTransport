# ---------------------------------------------------------------------------
# Transport-binary met driver
#
# Clean `src_v2` interface for preprocessed transport binaries:
#   - drivers own file/window timing and forcing I/O
#   - models own prognostic tracer state
#   - transport windows carry typed mass/flux forcing for one met window
# ---------------------------------------------------------------------------

"""
    AbstractTransportWindow{Basis}

Typed forcing window for one transport interval.

A transport window carries the preprocessed mass/flux forcing that the runtime
needs for one met interval. It does not own tracer state.
"""
abstract type AbstractTransportWindow{Basis <: AbstractMassBasis} end

struct StructuredFluxDeltas{AAm, ABm, ACm, AM}
    dam :: AAm
    dbm :: ABm
    dcm :: ACm
    dm  :: AM
end

struct FaceIndexedFluxDeltas{AH, ACm, AM}
    dhflux :: AH
    dcm    :: ACm
    dm     :: AM
end

struct StructuredTransportWindow{Basis <: AbstractMassBasis, M, PS, F, Q, D} <: AbstractTransportWindow{Basis}
    air_mass         :: M
    surface_pressure :: PS
    fluxes           :: F
    qv_start         :: Q
    qv_end           :: Q
    deltas           :: D
end

struct FaceIndexedTransportWindow{Basis <: AbstractMassBasis, M, PS, F, Q, D} <: AbstractTransportWindow{Basis}
    air_mass         :: M
    surface_pressure :: PS
    fluxes           :: F
    qv_start         :: Q
    qv_end           :: Q
    deltas           :: D
end

function Adapt.adapt_structure(to, deltas::StructuredFluxDeltas)
    dam = Adapt.adapt(to, deltas.dam)
    dbm = Adapt.adapt(to, deltas.dbm)
    dcm = Adapt.adapt(to, deltas.dcm)
    dm = Adapt.adapt(to, deltas.dm)
    return StructuredFluxDeltas{typeof(dam), typeof(dbm), typeof(dcm), typeof(dm)}(dam, dbm, dcm, dm)
end

function Adapt.adapt_structure(to, deltas::FaceIndexedFluxDeltas)
    dhflux = Adapt.adapt(to, deltas.dhflux)
    dcm = Adapt.adapt(to, deltas.dcm)
    dm = Adapt.adapt(to, deltas.dm)
    return FaceIndexedFluxDeltas{typeof(dhflux), typeof(dcm), typeof(dm)}(dhflux, dcm, dm)
end

function Adapt.adapt_structure(to, window::StructuredTransportWindow{B}) where {B <: AbstractMassBasis}
    air_mass = Adapt.adapt(to, window.air_mass)
    surface_pressure = Adapt.adapt(to, window.surface_pressure)
    fluxes = Adapt.adapt(to, window.fluxes)
    qv_start = Adapt.adapt(to, window.qv_start)
    qv_end = Adapt.adapt(to, window.qv_end)
    deltas = Adapt.adapt(to, window.deltas)
    return StructuredTransportWindow{B, typeof(air_mass), typeof(surface_pressure), typeof(fluxes), typeof(qv_start), typeof(deltas)}(
        air_mass, surface_pressure, fluxes, qv_start, qv_end, deltas)
end

function Adapt.adapt_structure(to, window::FaceIndexedTransportWindow{B}) where {B <: AbstractMassBasis}
    air_mass = Adapt.adapt(to, window.air_mass)
    surface_pressure = Adapt.adapt(to, window.surface_pressure)
    fluxes = Adapt.adapt(to, window.fluxes)
    qv_start = Adapt.adapt(to, window.qv_start)
    qv_end = Adapt.adapt(to, window.qv_end)
    deltas = Adapt.adapt(to, window.deltas)
    return FaceIndexedTransportWindow{B, typeof(air_mass), typeof(surface_pressure), typeof(fluxes), typeof(qv_start), typeof(deltas)}(
        air_mass, surface_pressure, fluxes, qv_start, qv_end, deltas)
end

mass_basis(::AbstractTransportWindow{B}) where {B} = B()
has_humidity_endpoints(window::AbstractTransportWindow) = window.qv_start !== nothing && window.qv_end !== nothing
has_flux_delta(window::AbstractTransportWindow) = window.deltas !== nothing

function StructuredTransportWindow(air_mass, surface_pressure, fluxes::StructuredFaceFluxState{B};
                                   qv_start = nothing, qv_end = nothing, deltas = nothing) where {B <: AbstractMassBasis}
    return StructuredTransportWindow{B, typeof(air_mass), typeof(surface_pressure), typeof(fluxes), typeof(qv_start), typeof(deltas)}(
        air_mass, surface_pressure, fluxes, qv_start, qv_end, deltas)
end

function FaceIndexedTransportWindow(air_mass, surface_pressure, fluxes::FaceIndexedFluxState{B};
                                    qv_start = nothing, qv_end = nothing, deltas = nothing) where {B <: AbstractMassBasis}
    return FaceIndexedTransportWindow{B, typeof(air_mass), typeof(surface_pressure), typeof(fluxes), typeof(qv_start), typeof(deltas)}(
        air_mass, surface_pressure, fluxes, qv_start, qv_end, deltas)
end

"""
    TransportBinaryDriver

Standalone `src_v2` met driver backed by a topology-generic transport binary.
"""
struct TransportBinaryDriver{FT, ReaderT, GridT} <: AbstractMassFluxMetDriver
    reader :: ReaderT
    grid   :: GridT
end

@inline function _cm_interface_mass(m, i, j, k, Nz)
    if k <= 1
        return m[i, j, 1]
    elseif k > Nz
        return m[i, j, Nz]
    else
        return max(m[i, j, k - 1], m[i, j, k])
    end
end

@inline function _cm_interface_mass(m, c, k, Nz)
    if k <= 1
        return m[c, 1]
    elseif k > Nz
        return m[c, Nz]
    else
        return max(m[c, k - 1], m[c, k])
    end
end

function _window_max_rel_cm(m::AbstractArray{FT, 3}, fluxes::StructuredFaceFluxState) where FT
    Nz = size(m, 3)
    worst = 0.0
    @inbounds for k in 1:size(fluxes.cm, 3), j in 1:size(fluxes.cm, 2), i in 1:size(fluxes.cm, 1)
        denom = max(Float64(_cm_interface_mass(m, i, j, k, Nz)), floatmin(Float64))
        ratio = abs(Float64(fluxes.cm[i, j, k])) / denom
        worst = max(worst, ratio)
    end
    return worst
end

function _window_max_rel_cm(m::AbstractMatrix{FT}, fluxes::FaceIndexedFluxState) where FT
    Nz = size(m, 2)
    worst = 0.0
    @inbounds for k in 1:size(fluxes.cm, 2), c in 1:size(fluxes.cm, 1)
        denom = max(Float64(_cm_interface_mass(m, c, k, Nz)), floatmin(Float64))
        ratio = abs(Float64(fluxes.cm[c, k])) / denom
        worst = max(worst, ratio)
    end
    return worst
end

function _validate_window_cm_sanity(reader::TransportBinaryReader; max_rel_cm::Real=1e-8)
    threshold = Float64(max_rel_cm)
    worst_ratio = 0.0
    worst_window = 0

    for win in 1:window_count(reader)
        m, _ps, fluxes = load_window!(reader, win)
        ratio = _window_max_rel_cm(m, fluxes)
        if ratio > worst_ratio
            worst_ratio = ratio
            worst_window = win
        end
        if ratio > threshold
            throw(ArgumentError(
                "TransportBinaryDriver sanity check failed for $(basename(reader.path)) " *
                "at window $win: max(abs(cm)/m)=$(ratio) exceeds threshold $(threshold). " *
                "This transport binary likely has inconsistent vertical mass fluxes; regenerate it " *
                "with the fixed Poisson-balanced preprocessor or disable validation explicitly."
            ))
        end
    end

    return worst_window, worst_ratio
end

function _validate_runtime_semantics(reader::TransportBinaryReader)
    h = reader.header
    expected_poisson_scale = 1.0 / (2 * h.steps_per_window)
    expected_poisson_semantics = "forward_window_mass_difference / (2 * steps_per_window)"

    h.flux_kind === :substep_mass_amount ||
        throw(ArgumentError("TransportBinaryDriver requires flux_kind = :substep_mass_amount, got $(h.flux_kind)"))

    h.air_mass_sampling === :window_start_endpoint ||
        throw(ArgumentError("TransportBinaryDriver requires air_mass_sampling = :window_start_endpoint, got $(h.air_mass_sampling)"))

    if has_flux_delta(reader)
        h.flux_sampling in (:window_start_endpoint, :window_constant) ||
            throw(ArgumentError("TransportBinaryDriver requires flux_sampling = :window_start_endpoint or :window_constant when deltas are present, got $(h.flux_sampling)"))
        h.delta_semantics === :forward_window_endpoint_difference ||
            throw(ArgumentError("TransportBinaryDriver requires delta_semantics = :forward_window_endpoint_difference, got $(h.delta_semantics)"))
    else
        h.flux_sampling in (:window_start_endpoint, :window_mean, :window_constant) ||
            throw(ArgumentError("TransportBinaryDriver supports flux_sampling = :window_start_endpoint, :window_mean, or :window_constant without deltas, got $(h.flux_sampling)"))
    end

    if has_qv_endpoints(reader)
        h.humidity_sampling === :window_endpoints ||
            throw(ArgumentError("TransportBinaryDriver requires humidity_sampling = :window_endpoints when qv_start/qv_end are present, got $(h.humidity_sampling)"))
    elseif has_qv(reader)
        h.humidity_sampling in (:single_field, :none) ||
            throw(ArgumentError("TransportBinaryDriver only supports humidity_sampling = :single_field for legacy qv payloads, got $(h.humidity_sampling)"))
    end

    if has_flux_delta(reader)
        poisson_scale = h.poisson_balance_target_scale
        if !isfinite(poisson_scale)
            h.format_version == 1 ||
                throw(ArgumentError("TransportBinaryDriver requires poisson_balance_target_scale metadata for delta-bearing transport binaries"))
            @warn "Legacy transport binary $(basename(reader.path)) is missing poisson_balance_target_scale; assuming $(expected_poisson_scale)"
            poisson_scale = expected_poisson_scale
        end

        poisson_semantics = h.poisson_balance_target_semantics
        if isempty(poisson_semantics)
            h.format_version == 1 ||
                throw(ArgumentError("TransportBinaryDriver requires poisson_balance_target_semantics metadata for delta-bearing transport binaries"))
            @warn "Legacy transport binary $(basename(reader.path)) is missing poisson_balance_target_semantics; assuming $(repr(expected_poisson_semantics))"
            poisson_semantics = expected_poisson_semantics
        end

        isapprox(poisson_scale, expected_poisson_scale; atol=eps(Float64)*8, rtol=0.0) ||
            throw(ArgumentError("TransportBinaryDriver requires poisson_balance_target_scale=$(expected_poisson_scale), got $(poisson_scale)"))
        poisson_semantics == expected_poisson_semantics ||
            throw(ArgumentError("TransportBinaryDriver requires poisson_balance_target_semantics = '$(expected_poisson_semantics)', got $(repr(poisson_semantics))"))
    end

    return nothing
end

function Base.summary(driver::TransportBinaryDriver{FT}) where {FT}
    return string(
        "TransportBinaryDriver{", FT, "}(",
        basename(driver.reader.path), ", ", grid_type(driver.reader), "/", horizontal_topology(driver.reader), ")"
    )
end

function Base.show(io::IO, driver::TransportBinaryDriver)
    reader = driver.reader
    h = reader.header
    print(io, summary(driver), "\n",
          "├── grid:          ", summary(driver.grid.horizontal), "\n",
          "├── basis:         ", air_mass_basis(driver), "\n",
          "├── timing:        dt=", window_dt(driver), " s, steps/window=", steps_per_window(driver), "\n",
          "├── payload:       ", join(String.(h.payload_sections), ", "), "\n",
          "├── humidity:      ", has_qv_endpoints(reader) ? "qv_start/qv_end" : (has_qv(reader) ? "qv" : "none"), "\n",
          "├── semantics:     air_mass=", h.air_mass_sampling, ", flux=", h.flux_sampling, "/", h.flux_kind, "\n",
          "└── windows:       ", total_windows(driver))
end

function TransportBinaryDriver(path::AbstractString;
                               FT::Type{<:AbstractFloat} = Float64,
                               arch = CPU(),
                               validate_windows::Bool = true,
                               max_rel_cm::Real = 1e-8)
    reader = TransportBinaryReader(String(path); FT=FT)
    _validate_runtime_semantics(reader)
    validate_windows && _validate_window_cm_sanity(reader; max_rel_cm=max_rel_cm)
    grid = load_grid(reader; FT=FT, arch=arch)
    return TransportBinaryDriver{FT, typeof(reader), typeof(grid)}(reader, grid)
end

Base.close(driver::TransportBinaryDriver) = close(driver.reader)

total_windows(driver::TransportBinaryDriver) = window_count(driver.reader)
window_dt(driver::TransportBinaryDriver{FT}) where {FT} = FT(driver.reader.header.dt_met_seconds)
steps_per_window(driver::TransportBinaryDriver) = driver.reader.header.steps_per_window
air_mass_basis(driver::TransportBinaryDriver) = mass_basis(driver.reader)
supports_moisture(driver::TransportBinaryDriver) = has_qv(driver.reader)
supports_native_vertical_flux(::TransportBinaryDriver) = true
driver_grid(driver::TransportBinaryDriver) = driver.grid
flux_interpolation_mode(driver::TransportBinaryDriver) =
    has_flux_delta(driver.reader) && driver.reader.header.flux_sampling !== :window_constant ? :interpolate : :constant

@inline function _interpolate_field!(dest, base, delta, λ)
    @. dest = base + λ * delta
    return dest
end

@inline function copy_fluxes!(dest::StructuredFaceFluxState, src::StructuredFaceFluxState)
    copyto!(dest.am, src.am)
    copyto!(dest.bm, src.bm)
    copyto!(dest.cm, src.cm)
    return dest
end

@inline function copy_fluxes!(dest::FaceIndexedFluxState, src::FaceIndexedFluxState)
    copyto!(dest.horizontal_flux, src.horizontal_flux)
    copyto!(dest.cm, src.cm)
    return dest
end

function interpolate_fluxes!(dest::StructuredFaceFluxState, window::StructuredTransportWindow, λ::Real)
    λ_ft = convert(eltype(dest.am), λ)
    if window.deltas === nothing
        return copy_fluxes!(dest, window.fluxes)
    end

    _interpolate_field!(dest.am, window.fluxes.am, window.deltas.dam, λ_ft)
    _interpolate_field!(dest.bm, window.fluxes.bm, window.deltas.dbm, λ_ft)
    _interpolate_field!(dest.cm, window.fluxes.cm, window.deltas.dcm, λ_ft)
    return dest
end

function interpolate_fluxes!(dest::FaceIndexedFluxState, window::FaceIndexedTransportWindow, λ::Real)
    λ_ft = convert(eltype(dest.horizontal_flux), λ)
    if window.deltas === nothing
        return copy_fluxes!(dest, window.fluxes)
    end

    _interpolate_field!(dest.horizontal_flux, window.fluxes.horizontal_flux, window.deltas.dhflux, λ_ft)
    _interpolate_field!(dest.cm, window.fluxes.cm, window.deltas.dcm, λ_ft)
    return dest
end

function expected_air_mass!(dest, window::AbstractTransportWindow, λ::Real)
    if window.deltas === nothing
        copyto!(dest, window.air_mass)
        return dest
    end
    λ_ft = convert(eltype(dest), λ)
    _interpolate_field!(dest, window.air_mass, window.deltas.dm, λ_ft)
    return dest
end

function interpolate_qv!(dest, window::AbstractTransportWindow, λ::Real)
    has_humidity_endpoints(window) || throw(ArgumentError("transport window does not carry qv_start/qv_end"))
    λ_ft = convert(eltype(dest), λ)
    @. dest = window.qv_start + λ_ft * (window.qv_end - window.qv_start)
    return dest
end

function _make_transport_window(m, ps, fluxes::StructuredFaceFluxState; qv_pair = nothing, deltas = nothing)
    delta_obj = if deltas === nothing
        nothing
    else
        haskey(deltas, :dam) || throw(ArgumentError("structured transport deltas require `dam`"))
        haskey(deltas, :dbm) || throw(ArgumentError("structured transport deltas require `dbm`"))
        haskey(deltas, :dcm) || throw(ArgumentError("structured transport deltas require `dcm`"))
        haskey(deltas, :dm) || throw(ArgumentError("structured transport deltas require `dm`"))
        StructuredFluxDeltas(deltas.dam, deltas.dbm, deltas.dcm, deltas.dm)
    end
    return StructuredTransportWindow(m, ps, fluxes;
                                     qv_start = qv_pair === nothing ? nothing : qv_pair.qv_start,
                                     qv_end = qv_pair === nothing ? nothing : qv_pair.qv_end,
                                     deltas = delta_obj)
end

function _make_transport_window(m, ps, fluxes::FaceIndexedFluxState; qv_pair = nothing, deltas = nothing)
    delta_obj = if deltas === nothing
        nothing
    else
        haskey(deltas, :dhflux) || throw(ArgumentError("face-indexed transport deltas require `dhflux`"))
        haskey(deltas, :dcm) || throw(ArgumentError("face-indexed transport deltas require `dcm`"))
        haskey(deltas, :dm) || throw(ArgumentError("face-indexed transport deltas require `dm`"))
        FaceIndexedFluxDeltas(deltas.dhflux, deltas.dcm, deltas.dm)
    end
    return FaceIndexedTransportWindow(m, ps, fluxes;
                                      qv_start = qv_pair === nothing ? nothing : qv_pair.qv_start,
                                      qv_end = qv_pair === nothing ? nothing : qv_pair.qv_end,
                                      deltas = delta_obj)
end

"""
    load_transport_window(driver, win)

Load one typed forcing window from the transport binary.
"""
function load_transport_window(driver::TransportBinaryDriver{FT, ReaderT, <:AtmosGrid{H}}, win::Int) where {FT, ReaderT, H <: AbstractStructuredMesh}
    m, ps, fluxes_any = load_window!(driver.reader, win)
    fluxes = fluxes_any::StructuredFaceFluxState
    qv_pair = load_qv_pair_window!(driver.reader, win)
    deltas = load_flux_delta_window!(driver.reader, win)
    return _make_transport_window(m, ps, fluxes; qv_pair=qv_pair, deltas=deltas)
end

function load_transport_window(driver::TransportBinaryDriver{FT, ReaderT, <:AtmosGrid{<:ReducedGaussianMesh}}, win::Int) where {FT, ReaderT}
    m, ps, fluxes_any = load_window!(driver.reader, win)
    fluxes = fluxes_any::FaceIndexedFluxState
    qv_pair = load_qv_pair_window!(driver.reader, win)
    deltas = load_flux_delta_window!(driver.reader, win)
    return _make_transport_window(m, ps, fluxes; qv_pair=qv_pair, deltas=deltas)
end

export AbstractTransportWindow
export StructuredFluxDeltas, FaceIndexedFluxDeltas
export StructuredTransportWindow, FaceIndexedTransportWindow
export TransportBinaryDriver, driver_grid, air_mass_basis, load_transport_window
export has_humidity_endpoints, interpolate_fluxes!, expected_air_mass!, interpolate_qv!, copy_fluxes!
