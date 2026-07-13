# ---------------------------------------------------------------------------
# Cubed-sphere transport driver
#
# The cubed-sphere runtime is panel-native instead of being forced through
# the flat structured transport-binary contract.
# ---------------------------------------------------------------------------

"""
    CubedSphereTransportWindow

One decoded v4 forcing window with panel-native air mass, pressure, face
fluxes, replay deltas, convection, surface
forcing, VDIFF fields, and exact TM5 diffusion exchange. Canonical CS binaries
do not carry humidity payloads.
"""
struct CubedSphereTransportWindow{Basis <: AbstractMassBasis, M, PS, F, Q, D, C, S, V, DK} <: AbstractTransportWindow{Basis}
    air_mass         :: M
    surface_pressure :: PS
    fluxes           :: F
    qv_start         :: Q
    qv_end           :: Q
    deltas           :: D
    convection       :: C
    surface          :: S
    vdiff            :: V
    dkg              :: DK  # exact TM5 interface exchange [kg s⁻¹], or nothing
end

struct CubedSphereFluxDeltas{AM}
    dm :: AM
end

function CubedSphereTransportWindow(air_mass, surface_pressure,
                                    fluxes::CubedSphereFaceFluxState{B};
                                    qv_start = nothing, qv_end = nothing,
                                    deltas = nothing, convection = nothing,
                                    surface = nothing, vdiff = nothing,
                                    dkg = nothing) where {B <: AbstractMassBasis}
    return CubedSphereTransportWindow{B, typeof(air_mass), typeof(surface_pressure), typeof(fluxes),
                                      typeof(qv_start), typeof(deltas), typeof(convection), typeof(surface),
                                      typeof(vdiff), typeof(dkg)}(
        air_mass, surface_pressure, fluxes, qv_start, qv_end, deltas, convection, surface, vdiff, dkg)
end

function Adapt.adapt_structure(to, window::CubedSphereTransportWindow{B}) where {B <: AbstractMassBasis}
    air_mass = Adapt.adapt(to, window.air_mass)
    surface_pressure = Adapt.adapt(to, window.surface_pressure)
    fluxes = Adapt.adapt(to, window.fluxes)
    qv_start = Adapt.adapt(to, window.qv_start)
    qv_end = Adapt.adapt(to, window.qv_end)
    deltas = Adapt.adapt(to, window.deltas)
    convection = Adapt.adapt(to, window.convection)
    surface = Adapt.adapt(to, window.surface)
    vdiff = Adapt.adapt(to, window.vdiff)
    dkg = Adapt.adapt(to, window.dkg)
    return CubedSphereTransportWindow{B, typeof(air_mass), typeof(surface_pressure), typeof(fluxes),
                                      typeof(qv_start), typeof(deltas), typeof(convection), typeof(surface),
                                      typeof(vdiff), typeof(dkg)}(
        air_mass, surface_pressure, fluxes, qv_start, qv_end, deltas, convection, surface, vdiff, dkg)
end

function Adapt.adapt_structure(to, deltas::CubedSphereFluxDeltas)
    return CubedSphereFluxDeltas(Adapt.adapt(to, deltas.dm))
end

"""
    CubedSphereTransportDriver(path; FT=Float64, arch=CPU(), Hp=1)

Runtime driver for canonical v4 cubed-sphere binaries. It owns a validated
reader and reconstructs the AtmosGrid used to allocate panel-native state and
operator workspaces.
"""
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
          "├── timing:        dt=", window_dt(driver), " s, steps/window=",
              _steps_per_window_summary(steps_per_window(driver), steps_per_window_schedule(driver)), "\n",
          "└── windows:       ", total_windows(driver))
end

function uses_binary_substep_contract(driver::CubedSphereTransportDriver)
    hdr = driver.reader.header.raw_header
    contract = get(hdr, "runtime_substep_contract", nothing)
    return contract == "binary_schedule"
end

function _validate_replay_consistency_cs(
    reader::TransportBinaryReader{FT, DiskFT, CubedSphereBinaryGeometry},
) where {FT, DiskFT}
    if get(ENV, "ATMOSTR_NO_REPLAY_CHECK", "0") == "1"
        return nothing
    end
    tol_rel = replay_tolerance(FT)
    Nt = window_count(reader)
    Nt >= 1 || return nothing

    worst_rel = 0.0
    worst_abs = 0.0
    worst_win = 0
    worst_idx = (0, 0, 0, 0)

    for k in 1:Nt
        cur = load_window!(reader, k)
        steps = reader.header.steps_per_window_by_window[k]
        if flux_kind(reader) === :full_window_mass_amount
            scale = FT(1) / FT(2 * steps)
            _scale_cs_replay_panels!(cur.am, scale)
            _scale_cs_replay_panels!(cur.bm, scale)
            _scale_cs_replay_panels!(cur.cm, scale)
        end
        m_target = if k < Nt
            load_window!(reader, k + 1).m
        elseif has_flux_delta(reader)
            deltas = load_flux_delta_window!(reader, k)
            if deltas === nothing || !haskey(deltas, :dm)
                cur.m
            else
                ntuple(p -> cur.m[p] .+ deltas.dm[p], length(cur.m))
            end
        else
            cur.m
        end
        diag = verify_window_continuity_cs(cur.m, cur.am, cur.bm, cur.cm, m_target, steps)
        if diag.max_rel_err > worst_rel
            worst_rel = diag.max_rel_err
            worst_abs = diag.max_abs_err
            worst_win = k
            worst_idx = diag.worst_idx
        end
    end

    worst_rel <= tol_rel ||
        throw(ArgumentError(
            "CubedSphereTransportDriver replay-consistency gate FAILED for " *
            "$(basename(reader.path)): rel=$(worst_rel) > tol=$(tol_rel) at window " *
            "$worst_win cell $worst_idx (abs=$worst_abs kg). Stored CS fluxes do not " *
            "integrate to the stored mass target under palindrome continuity. " *
            "Regenerate the binary with the CS replay-safe preprocessor or bypass " *
            "with ENV[\"ATMOSTR_NO_REPLAY_CHECK\"]=\"1\" for diagnostic runs."
        ))

    @info "Replay continuity gate passed: $(basename(reader.path)) " *
          "topology=cubed_sphere worst_rel=$(worst_rel) worst_window=$(worst_win)"
    return (worst_window = worst_win, worst_rel = worst_rel, worst_abs = worst_abs)
end

@inline function _scale_cs_replay_panels!(panels::NTuple{6}, scale)
    @inbounds for p in 1:6
        panels[p] .*= scale
    end
    return panels
end

function CubedSphereTransportDriver(
    reader::TransportBinaryReader{FT, DiskFT, CubedSphereBinaryGeometry};
    arch = CPU(), Hp::Int = 1,
) where {FT, DiskFT}
    grid = load_grid(reader; FT=FT, arch=arch, Hp=Hp)
    return CubedSphereTransportDriver{FT, typeof(reader), typeof(grid)}(reader, grid)
end

function CubedSphereTransportDriver(path::AbstractString;
                                    FT::Type{<:AbstractFloat} = Float64,
                                    arch = CPU(),
                                    Hp::Int = 1,
                                    validate_replay::Bool = false)
    reader = TransportBinaryReader(String(path); FT=FT)
    grid_type(reader) === :cubed_sphere || throw(ArgumentError(
        "CubedSphereTransportDriver requires a cubed-sphere binary; got $(grid_type(reader))"))
    replay_on = validate_replay || get(ENV, "ATMOSTR_REPLAY_CHECK", "0") == "1"
    replay_on && _validate_replay_consistency_cs(reader)
    return CubedSphereTransportDriver(reader; arch=arch, Hp=Hp)
end

total_windows(driver::CubedSphereTransportDriver) = window_count(driver.reader)
window_dt(driver::CubedSphereTransportDriver{FT}) where {FT} = FT(driver.reader.header.dt_met_seconds)
steps_per_window(driver::CubedSphereTransportDriver) = driver.reader.header.steps_per_window
steps_per_window(driver::CubedSphereTransportDriver, win::Integer) =
    driver.reader.header.steps_per_window_by_window[Int(win)]
steps_per_window_schedule(driver::CubedSphereTransportDriver) =
    copy(driver.reader.header.steps_per_window_by_window)
air_mass_basis(driver::CubedSphereTransportDriver) = mass_basis(driver.reader)
supports_native_vertical_flux(::CubedSphereTransportDriver) = true
supports_moisture(::CubedSphereTransportDriver) = false
supports_convection(driver::CubedSphereTransportDriver) =
    has_cmfmc(driver.reader) || has_tm5_convection(driver.reader)
supports_diffusion(driver::CubedSphereTransportDriver) =
    has_surface(driver.reader) ||
    :dkg in driver.reader.header.payload_sections
driver_grid(driver::CubedSphereTransportDriver) = driver.grid
flux_interpolation_mode(::CubedSphereTransportDriver) = :constant
flux_kind(driver::CubedSphereTransportDriver) = flux_kind(driver.reader)

Base.close(driver::CubedSphereTransportDriver) = close(driver.reader)

"""
    release_payload!(driver::CubedSphereTransportDriver)

Hint to the OS that the driver's memory-mapped binary payload is no longer
needed, so the kernel drops the faulted file-cache pages now instead of holding
them — charged to the process memory cgroup — until reclaim pressure. The per-day
CS loop rebuilds a fresh driver each day; without this a long run accumulates
every day's mmap as cgroup-charged `inactive_file`, leaving no cgroup headroom
for the user's other processes (on a per-user cgroup the run can starve them).
`munmap`/`finalize` does NOT achieve this — clean file pages survive the unmap —
but `madvise(MADV_DONTNEED)` evicts them immediately (measured: ~32 GB → 0.7 GB
for one ERA5 C180/L137 day).

Safe: the mapping is read-only and file-backed, so any later access simply
re-faults from disk rather than crashing — `madvise` here is purely an
optimization. Linux-only; a no-op on other platforms or on `madvise` failure.
Call once the day's windows have been consumed (the runner calls it right after
the day's `close(driver)`); the binary writer page-aligns the payload, but the
start is rounded down to a page boundary defensively.
"""
function release_payload!(driver::CubedSphereTransportDriver)
    Sys.islinux() || return nothing
    data = driver.reader.data
    (data isa Array && !isempty(data)) || return nothing
    MADV_DONTNEED = Cint(4)                      # Linux <sys/mman.h>
    pg   = ccall(:getpagesize, Cint, ())
    addr = UInt(pointer(data))
    base = addr & ~(UInt(pg) - 1)                # round start down to a page boundary
    len  = Csize_t(sizeof(data) + (addr - base)) # extend length to cover the rounding
    ccall(:madvise, Cint, (Ptr{Cvoid}, Csize_t, Cint), Ptr{Cvoid}(base), len, MADV_DONTNEED)
    return nothing
end

@inline _cs_basis_type(
    reader::TransportBinaryReader{<:Any, <:Any, CubedSphereBinaryGeometry},
) =
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
    _copy_cs_storage!(dest, window.air_mass)
    window.deltas === nothing && return dest
    λ_ft = convert(eltype(dest[1]), λ)
    @inbounds for p in 1:6
        @. dest[p] = dest[p] + λ_ft * window.deltas.dm[p]
    end
    return dest
end

function load_transport_window(driver::CubedSphereTransportDriver, win::Int)
    raw = load_window!(driver.reader, win)
    Hp = driver.grid.horizontal.Hp
    panels_m = ntuple(p -> _pad_horizontal(raw.m[p], Hp), 6)
    panels_ps = raw.ps
    panels_am = ntuple(p -> _pad_horizontal(raw.am[p], Hp), 6)
    panels_bm = ntuple(p -> _pad_horizontal(raw.bm[p], Hp), 6)
    panels_cm = ntuple(p -> _pad_horizontal(raw.cm[p], Hp), 6)
    basis = _cs_basis_type(driver.reader)
    fluxes = CubedSphereFaceFluxState{basis}(panels_am, panels_bm, panels_cm)
    # `raw.tm5_fields` is a NamedTuple of per-panel
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
    delta_raw = load_flux_delta_window!(driver.reader, win)
    deltas = delta_raw === nothing ? nothing : CubedSphereFluxDeltas(
        ntuple(p -> _pad_horizontal(delta_raw.dm[p], Hp), 6))
    return CubedSphereTransportWindow(panels_m, panels_ps, fluxes;
                                      deltas = deltas,
                                      convection = convection,
                                      surface = raw.surface,
                                      vdiff = raw.vdiff,
                                      dkg = raw.dkg)
end

export CubedSphereTransportWindow, CubedSphereTransportDriver
