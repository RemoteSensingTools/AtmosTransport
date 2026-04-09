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

function TransportBinaryDriver(path::AbstractString;
                               FT::Type{<:AbstractFloat} = Float64,
                               arch = CPU())
    reader = TransportBinaryReader(String(path); FT=FT)
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

function load_transport_window(driver::TransportBinaryDriver{FT, ReaderT, <:AtmosGrid{ReducedGaussianMesh}}, win::Int) where {FT, ReaderT}
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
