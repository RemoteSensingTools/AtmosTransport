# ---------------------------------------------------------------------------
# Cubed-sphere transport driver
#
# Plan 22B keeps the cubed-sphere runtime panel-native instead of forcing it
# through the flat structured transport-binary contract.
# ---------------------------------------------------------------------------

struct CubedSphereTransportWindow{Basis <: AbstractMassBasis, M, PS, F, Q, D, C} <: AbstractTransportWindow{Basis}
    air_mass         :: M
    surface_pressure :: PS
    fluxes           :: F
    qv_start         :: Q
    qv_end           :: Q
    deltas           :: D
    convection       :: C
end

function CubedSphereTransportWindow(air_mass, surface_pressure,
                                    fluxes::CubedSphereFaceFluxState{B};
                                    qv_start = nothing, qv_end = nothing,
                                    deltas = nothing, convection = nothing) where {B <: AbstractMassBasis}
    return CubedSphereTransportWindow{B, typeof(air_mass), typeof(surface_pressure), typeof(fluxes),
                                      typeof(qv_start), typeof(deltas), typeof(convection)}(
        air_mass, surface_pressure, fluxes, qv_start, qv_end, deltas, convection)
end

function Adapt.adapt_structure(to, window::CubedSphereTransportWindow{B}) where {B <: AbstractMassBasis}
    air_mass = Adapt.adapt(to, window.air_mass)
    surface_pressure = Adapt.adapt(to, window.surface_pressure)
    fluxes = Adapt.adapt(to, window.fluxes)
    qv_start = Adapt.adapt(to, window.qv_start)
    qv_end = Adapt.adapt(to, window.qv_end)
    deltas = Adapt.adapt(to, window.deltas)
    convection = Adapt.adapt(to, window.convection)
    return CubedSphereTransportWindow{B, typeof(air_mass), typeof(surface_pressure), typeof(fluxes),
                                      typeof(qv_start), typeof(deltas), typeof(convection)}(
        air_mass, surface_pressure, fluxes, qv_start, qv_end, deltas, convection)
end

struct CubedSphereTransportDriver{FT, ReaderT, GridT} <: AbstractMassFluxMetDriver
    reader :: ReaderT
    grid   :: GridT
end

Base.summary(driver::CubedSphereTransportDriver{FT}) where {FT} =
    string("CubedSphereTransportDriver{", FT, "}(", basename(driver.reader.path), ", ", driver.reader.header.nwindow, " windows)")

function Base.show(io::IO, driver::CubedSphereTransportDriver)
    print(io, summary(driver), "\n",
          "├── grid:          C", driver.grid.horizontal.Nc, ", Hp=", driver.grid.horizontal.Hp, "\n",
          "├── basis:         ", air_mass_basis(driver), "\n",
          "├── timing:        dt=", window_dt(driver), " s, steps/window=", steps_per_window(driver), "\n",
          "└── windows:       ", total_windows(driver))
end

window_count(reader::CubedSphereBinaryReader) = cs_window_count(reader)
mass_basis(reader::CubedSphereBinaryReader) = reader.header.mass_basis
grid_type(::CubedSphereBinaryReader) = :cubed_sphere
horizontal_topology(::CubedSphereBinaryReader) = :structureddirectional
A_ifc(reader::CubedSphereBinaryReader) = reader.header.A_ifc
B_ifc(reader::CubedSphereBinaryReader) = reader.header.B_ifc
has_qv(::CubedSphereBinaryReader) = false
has_qv_endpoints(::CubedSphereBinaryReader) = false
has_flux_delta(::CubedSphereBinaryReader) = false
has_cmfmc(reader::CubedSphereBinaryReader) = :cmfmc in reader.header.payload_sections
has_tm5conv(reader::CubedSphereBinaryReader) =
    all(s in reader.header.payload_sections for s in (:entu, :detu, :entd, :detd))
has_tm5_convection(reader::CubedSphereBinaryReader) = has_tm5conv(reader)

function _cs_header_symbol(reader::CubedSphereBinaryReader, key::AbstractString, default::Symbol)
    value = get(reader.header.raw_header, key, String(default))
    return Symbol(replace(lowercase(String(value)), '-' => '_', ' ' => '_'))
end

source_flux_sampling(reader::CubedSphereBinaryReader) = _cs_header_symbol(reader, "source_flux_sampling", :window_start_endpoint)
air_mass_sampling(reader::CubedSphereBinaryReader) = _cs_header_symbol(reader, "air_mass_sampling", :window_start_endpoint)
flux_sampling(reader::CubedSphereBinaryReader) = _cs_header_symbol(reader, "flux_sampling", :window_constant)
flux_kind(reader::CubedSphereBinaryReader) = _cs_header_symbol(reader, "flux_kind", :substep_mass_amount)
humidity_sampling(reader::CubedSphereBinaryReader) = _cs_header_symbol(reader, "humidity_sampling", :none)
delta_semantics(reader::CubedSphereBinaryReader) = _cs_header_symbol(reader, "delta_semantics", :none)

Base.close(reader::CubedSphereBinaryReader) = close(reader.io)

function load_grid(reader::CubedSphereBinaryReader;
                   FT::Type{<:AbstractFloat} = Float64,
                   arch = CPU(),
                   Hp::Int = 1)
    vc = HybridSigmaPressure(FT.(A_ifc(reader)), FT.(B_ifc(reader)))
    mesh = CubedSphereMesh(; FT=FT, Nc=reader.header.Nc, Hp=Hp, convention=mesh_convention(reader))
    return AtmosGrid(mesh, vc, arch; FT=FT)
end

function CubedSphereTransportDriver(reader::CubedSphereBinaryReader{FT};
                                    arch = CPU(),
                                    Hp::Int = 1) where {FT}
    grid = load_grid(reader; FT=FT, arch=arch, Hp=Hp)
    return CubedSphereTransportDriver{FT, typeof(reader), typeof(grid)}(reader, grid)
end

function CubedSphereTransportDriver(path::AbstractString;
                                    FT::Type{<:AbstractFloat} = Float64,
                                    arch = CPU(),
                                    Hp::Int = 1)
    reader = CubedSphereBinaryReader(String(path); FT=FT)
    return CubedSphereTransportDriver(reader; arch=arch, Hp=Hp)
end

total_windows(driver::CubedSphereTransportDriver) = window_count(driver.reader)
window_dt(driver::CubedSphereTransportDriver{FT}) where {FT} = FT(driver.reader.header.dt_met_seconds)
steps_per_window(driver::CubedSphereTransportDriver) = driver.reader.header.steps_per_window
air_mass_basis(driver::CubedSphereTransportDriver) = mass_basis(driver.reader)
supports_native_vertical_flux(::CubedSphereTransportDriver) = true
supports_moisture(::CubedSphereTransportDriver) = false
supports_convection(driver::CubedSphereTransportDriver) =
    has_cmfmc(driver.reader) || has_tm5conv(driver.reader)
driver_grid(driver::CubedSphereTransportDriver) = driver.grid
flux_interpolation_mode(::CubedSphereTransportDriver) = :constant

Base.close(driver::CubedSphereTransportDriver) = close(driver.reader)

@inline _cs_basis_type(reader::CubedSphereBinaryReader) =
    mass_basis(reader) === :dry ? DryBasis : MoistBasis

@inline function _pad_horizontal(a::AbstractArray{T, N}, Hp::Int) where {T, N}
    dims = ntuple(d -> d <= 2 ? size(a, d) + 2 * Hp : size(a, d), N)
    padded = zeros(T, dims...)
    ranges = ntuple(d -> d <= 2 ? ((Hp + 1):(Hp + size(a, d))) : axes(a, d), N)
    padded[ranges...] .= a
    return padded
end

@inline function _copy_cs_storage!(dest::NTuple{6}, src::NTuple{6})
    @inbounds for p in 1:6
        copyto!(dest[p], src[p])
    end
    return dest
end

@inline function copy_fluxes!(dest::CubedSphereFaceFluxState, src::CubedSphereFaceFluxState)
    _copy_cs_storage!(dest.am, src.am)
    _copy_cs_storage!(dest.bm, src.bm)
    _copy_cs_storage!(dest.cm, src.cm)
    return dest
end

function interpolate_fluxes!(dest::CubedSphereFaceFluxState, window::CubedSphereTransportWindow, λ::Real)
    return copy_fluxes!(dest, window.fluxes)
end

function expected_air_mass!(dest::NTuple{6}, window::CubedSphereTransportWindow, λ::Real)
    return _copy_cs_storage!(dest, window.air_mass)
end

function load_transport_window(driver::CubedSphereTransportDriver, win::Int)
    raw = load_cs_window(driver.reader, win)
    Hp = driver.grid.horizontal.Hp
    panels_m = ntuple(p -> _pad_horizontal(raw.m[p], Hp), 6)
    panels_ps = raw.ps
    panels_am = ntuple(p -> _pad_horizontal(raw.am[p], Hp), 6)
    panels_bm = ntuple(p -> _pad_horizontal(raw.bm[p], Hp), 6)
    panels_cm = ntuple(p -> _pad_horizontal(raw.cm[p], Hp), 6)
    basis = _cs_basis_type(driver.reader)
    fluxes = CubedSphereFaceFluxState{basis}(panels_am, panels_bm, panels_cm)
    # Plan 23 Commit 3: `raw.tm5_fields` is a NamedTuple of per-panel
    # NTuples `(entu, detu, entd, detd)` when the binary carries TM5
    # sections, or `nothing` otherwise. The runtime validator in
    # DrivenSimulation decides whether TM5Convection can run against
    # this forcing; constructing ConvectionForcing here is
    # capability-preserving (present stays present, absent stays
    # absent).
    has_cmfmc_fwd = raw.cmfmc !== nothing
    has_tm5_fwd   = raw.tm5_fields !== nothing
    convection = if has_cmfmc_fwd || has_tm5_fwd
        ConvectionForcing(raw.cmfmc, raw.dtrain, raw.tm5_fields)
    else
        nothing
    end
    return CubedSphereTransportWindow(panels_m, panels_ps, fluxes; convection = convection)
end

export CubedSphereTransportWindow, CubedSphereTransportDriver
