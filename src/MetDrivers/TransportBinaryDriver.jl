# ---------------------------------------------------------------------------
# Transport-binary met driver
#
# Clean `src` interface for preprocessed transport binaries:
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

struct StructuredTransportWindow{Basis <: AbstractMassBasis, M, PS, F, Q, D, C} <: AbstractTransportWindow{Basis}
    air_mass         :: M
    surface_pressure :: PS
    fluxes           :: F
    qv_start         :: Q
    qv_end           :: Q
    deltas           :: D
    convection       :: C   # ::Union{Nothing, ConvectionForcing} — plan 18 Commit 2
end

struct FaceIndexedTransportWindow{Basis <: AbstractMassBasis, M, PS, F, Q, D, C} <: AbstractTransportWindow{Basis}
    air_mass         :: M
    surface_pressure :: PS
    fluxes           :: F
    qv_start         :: Q
    qv_end           :: Q
    deltas           :: D
    convection       :: C   # ::Union{Nothing, ConvectionForcing} — plan 18 Commit 2
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
    convection = Adapt.adapt(to, window.convection)
    return StructuredTransportWindow{B, typeof(air_mass), typeof(surface_pressure), typeof(fluxes), typeof(qv_start), typeof(deltas), typeof(convection)}(
        air_mass, surface_pressure, fluxes, qv_start, qv_end, deltas, convection)
end

function Adapt.adapt_structure(to, window::FaceIndexedTransportWindow{B}) where {B <: AbstractMassBasis}
    air_mass = Adapt.adapt(to, window.air_mass)
    surface_pressure = Adapt.adapt(to, window.surface_pressure)
    fluxes = Adapt.adapt(to, window.fluxes)
    qv_start = Adapt.adapt(to, window.qv_start)
    qv_end = Adapt.adapt(to, window.qv_end)
    deltas = Adapt.adapt(to, window.deltas)
    convection = Adapt.adapt(to, window.convection)
    return FaceIndexedTransportWindow{B, typeof(air_mass), typeof(surface_pressure), typeof(fluxes), typeof(qv_start), typeof(deltas), typeof(convection)}(
        air_mass, surface_pressure, fluxes, qv_start, qv_end, deltas, convection)
end

mass_basis(::AbstractTransportWindow{B}) where {B} = B()
has_humidity_endpoints(window::AbstractTransportWindow) = window.qv_start !== nothing && window.qv_end !== nothing
has_flux_delta(window::AbstractTransportWindow) = window.deltas !== nothing

# Plan 18 Commit 2: window-level convection probe. Extends the
# `ConvectionForcing` overload from `ConvectionForcing.jl` to the
# window layer.
has_convection_forcing(window::AbstractTransportWindow) = window.convection !== nothing

function StructuredTransportWindow(air_mass, surface_pressure, fluxes::StructuredFaceFluxState{B};
                                   qv_start = nothing, qv_end = nothing, deltas = nothing,
                                   convection = nothing) where {B <: AbstractMassBasis}
    return StructuredTransportWindow{B, typeof(air_mass), typeof(surface_pressure), typeof(fluxes), typeof(qv_start), typeof(deltas), typeof(convection)}(
        air_mass, surface_pressure, fluxes, qv_start, qv_end, deltas, convection)
end

function FaceIndexedTransportWindow(air_mass, surface_pressure, fluxes::FaceIndexedFluxState{B};
                                    qv_start = nothing, qv_end = nothing, deltas = nothing,
                                    convection = nothing) where {B <: AbstractMassBasis}
    return FaceIndexedTransportWindow{B, typeof(air_mass), typeof(surface_pressure), typeof(fluxes), typeof(qv_start), typeof(deltas), typeof(convection)}(
        air_mass, surface_pressure, fluxes, qv_start, qv_end, deltas, convection)
end

"""
    TransportBinaryDriver

Standalone `src` met driver backed by a topology-generic transport binary.
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

@inline function _replay_window_pair(::StructuredDirectionalReplayLayout,
                                     div_scratch::AbstractArray{Float64, 3},
                                     m_cur::AbstractArray{FT, 3},
                                     fluxes::StructuredFaceFluxState,
                                     m_next::AbstractArray{FT, 3},
                                     steps_per_window::Integer) where FT
    return verify_window_continuity(structured_replay_layout(), div_scratch,
                                    m_cur, fluxes.cm, m_next, steps_per_window,
                                    fluxes.am, fluxes.bm)
end

"""
    _validate_replay_consistency_ll(reader::TransportBinaryReader)

Plan 39 Commit F — load-time replay gate for LL structured binaries.
Walks every consecutive window pair (k, k+1) and asserts

    m[k] − 2·steps·(∇·am + ∇·bm + ∂_k cm)  ≈  m[k+1]

to within `tol_rel = 1e-10` (Float64) / `1e-4` (Float32). This mirrors the
write-time gate from Commit E but fires at driver construction so a
binary produced by an older preprocessor (with the dry-basis Δb×pit cm
closure bug) is rejected before any runtime integration.

Bypass with env var `ATMOSTR_NO_REPLAY_CHECK=1`.
"""
function _validate_replay_consistency_ll(reader::TransportBinaryReader)
    if get(ENV, "ATMOSTR_NO_REPLAY_CHECK", "0") == "1"
        return nothing
    end
    FT = reader.header.float_type
    tol_rel = replay_tolerance(FT)
    steps = reader.header.steps_per_window
    Nt = window_count(reader)
    Nt >= 2 || return nothing

    m_cur, _ps_cur, fluxes = load_window!(reader, 1)
    div_scratch = Array{Float64}(undef, size(m_cur))
    layout = structured_replay_layout()
    worst_rel = 0.0
    worst_abs = 0.0
    worst_win = 0
    worst_idx = (0, 0, 0)
    for k in 1:(Nt - 1)
        m_next, _ps_next, _fluxes_next = load_window!(reader, k + 1)
        diag = _replay_window_pair(layout, div_scratch, m_cur, fluxes, m_next, steps)
        if diag.max_rel_err > worst_rel
            worst_rel = diag.max_rel_err
            worst_abs = diag.max_abs_err
            worst_win = k
            worst_idx = diag.worst_idx
        end
        # Advance: next window becomes current.
        m_cur = m_next
        _, _, fluxes = load_window!(reader, k + 1)
    end

    worst_rel <= tol_rel ||
        throw(ArgumentError(
            "TransportBinaryDriver replay-consistency gate FAILED for " *
            "$(basename(reader.path)): rel=$(worst_rel) > tol=$(tol_rel) at window " *
            "$worst_win cell $worst_idx (abs=$worst_abs kg). Stored fluxes do not " *
            "integrate to stored m_next under palindrome continuity. Regenerate the " *
            "binary with the plan-39 preprocessor fix (explicit-dm cm closure) or " *
            "bypass with ENV[\"ATMOSTR_NO_REPLAY_CHECK\"]=\"1\" for diagnostic runs."
        ))

    return (worst_window = worst_win, worst_rel = worst_rel, worst_abs = worst_abs)
end

# Dispatch stub: CS topology not yet covered by load-time replay.
# The write-time gate (Commit E) covers it; extend here if needed.
_validate_replay_consistency_ll(::Any) = nothing

@inline function _rg_face_connectivity(mesh)
    nf = Grids.nfaces(mesh)
    left  = Vector{Int32}(undef, nf)
    right = Vector{Int32}(undef, nf)
    @inbounds for f in 1:nf
        l, r = Grids.face_cells(mesh, f)
        left[f]  = Int32(l)
        right[f] = Int32(r)
    end
    return left, right
end

@inline function _replay_window_pair(layout::FaceIndexedReplayLayout,
                                     div_scratch::AbstractMatrix{Float64},
                                     m_cur::AbstractMatrix{FT},
                                     fluxes::FaceIndexedFluxState,
                                     m_next::AbstractMatrix{FT},
                                     steps_per_window::Integer) where FT
    return verify_window_continuity(layout, div_scratch,
                                    m_cur, fluxes.cm, m_next, steps_per_window,
                                    fluxes.horizontal_flux)
end

"""
    _validate_replay_consistency_rg(reader::TransportBinaryReader, grid)

Plan 39 Commit F — load-time replay gate for RG (`:faceindexed`) binaries.
Uses the `ReducedGaussianMesh` from `grid.horizontal` to build face-cell
connectivity, then walks consecutive window pairs and asserts

    m[k] − 2·steps·(div_face_flux + ∂_k cm) ≈ m[k+1]

to within `tol_rel = 1e-10` (Float64) / `1e-4` (Float32). Bypass with
`ENV["ATMOSTR_NO_REPLAY_CHECK"]="1"`.
"""
function _validate_replay_consistency_rg(reader::TransportBinaryReader, grid)
    if get(ENV, "ATMOSTR_NO_REPLAY_CHECK", "0") == "1"
        return nothing
    end
    FT = reader.header.float_type
    tol_rel = replay_tolerance(FT)
    steps = reader.header.steps_per_window
    Nt = window_count(reader)
    Nt >= 2 || return nothing

    face_left, face_right = _rg_face_connectivity(grid.horizontal)
    layout = faceindexed_replay_layout(face_left, face_right)

    m_cur, _, fluxes = load_window!(reader, 1)
    _, Nz = size(m_cur)
    div_scratch = zeros(Float64, size(m_cur, 1), Nz)
    worst_rel = 0.0
    worst_abs = 0.0
    worst_win = 0
    worst_idx = (0, 0)
    for k in 1:(Nt - 1)
        m_next, _, _ = load_window!(reader, k + 1)
        diag = _replay_window_pair(layout, div_scratch, m_cur, fluxes, m_next, steps)
        if diag.max_rel_err > worst_rel
            worst_rel = diag.max_rel_err
            worst_abs = diag.max_abs_err
            worst_win = k
            worst_idx = diag.worst_idx
        end
        m_cur = m_next
        _, _, fluxes = load_window!(reader, k + 1)
    end

    worst_rel <= tol_rel ||
        throw(ArgumentError(
            "TransportBinaryDriver replay-consistency gate FAILED for " *
            "$(basename(reader.path)): rel=$(worst_rel) > tol=$(tol_rel) at window " *
            "$worst_win cell $worst_idx (abs=$worst_abs kg). Stored fluxes do not " *
            "integrate to stored m_next under palindrome continuity. Regenerate the " *
            "binary with the plan-39 preprocessor fix (explicit-dm cm closure) or " *
            "bypass with ENV[\"ATMOSTR_NO_REPLAY_CHECK\"]=\"1\" for diagnostic runs."
        ))

    return (worst_window = worst_win, worst_rel = worst_rel, worst_abs = worst_abs)
end

_validate_replay_consistency_rg(::Any, ::Any) = nothing

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
                               validate_replay::Bool = false,
                               max_rel_cm::Real = 0.01)
    reader = TransportBinaryReader(String(path); FT=FT)
    _validate_runtime_semantics(reader)
    validate_windows && _validate_window_cm_sanity(reader; max_rel_cm=max_rel_cm)
    grid = load_grid(reader; FT=FT, arch=arch)
    # Plan 39 Commit F: load-time replay-consistency gate. Opt-in because
    # the write-time Commit E gate already guarantees continuity for
    # binaries we produce; the load-time gate is for suspect binaries
    # (manual imports, file corruption, older preprocessor versions).
    # Set `validate_replay=true` or `ENV["ATMOSTR_REPLAY_CHECK"]="1"` to
    # enable; disable the in-flight check with `ATMOSTR_NO_REPLAY_CHECK=1`.
    replay_on = validate_replay || get(ENV, "ATMOSTR_REPLAY_CHECK", "0") == "1"
    if replay_on
        topo = horizontal_topology(reader)
        if topo === :structureddirectional
            _validate_replay_consistency_ll(reader)
        elseif topo === :faceindexed
            _validate_replay_consistency_rg(reader, grid)
        end
    end
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

function _make_transport_window(m, ps, fluxes::StructuredFaceFluxState;
                                qv_pair = nothing, deltas = nothing,
                                convection = nothing)
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
                                     deltas = delta_obj,
                                     convection = convection)
end

function _make_transport_window(m, ps, fluxes::FaceIndexedFluxState;
                                qv_pair = nothing, deltas = nothing,
                                convection = nothing)
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
                                      deltas = delta_obj,
                                      convection = convection)
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
    convection = _load_transport_binary_convection_forcing(driver.reader, win)
    return _make_transport_window(m, ps, fluxes; qv_pair=qv_pair, deltas=deltas,
                                    convection=convection)
end

function load_transport_window(driver::TransportBinaryDriver{FT, ReaderT, <:AtmosGrid{<:ReducedGaussianMesh}}, win::Int) where {FT, ReaderT}
    m, ps, fluxes_any = load_window!(driver.reader, win)
    fluxes = fluxes_any::FaceIndexedFluxState
    qv_pair = load_qv_pair_window!(driver.reader, win)
    deltas = load_flux_delta_window!(driver.reader, win)
    convection = _load_transport_binary_convection_forcing(driver.reader, win)
    return _make_transport_window(m, ps, fluxes; qv_pair=qv_pair, deltas=deltas,
                                    convection=convection)
end

# Plan 23 Commit 3 — returns a ConvectionForcing with populated
# tm5_fields when the LL/RG binary carries TM5 sections; nothing
# otherwise.  CMFMC isn't yet written to LL/RG transport binaries
# (only CS), so cmfmc/dtrain stay nothing on this path.  The
# DrivenSimulation validator enforces the operator-forcing
# compatibility at runtime.
function _load_transport_binary_convection_forcing(reader::TransportBinaryReader, win::Int)
    has_tm5_convection(reader) || return nothing
    tm5 = load_tm5_convection_window!(reader, win)
    return ConvectionForcing(nothing, nothing, tm5)
end

export AbstractTransportWindow
export StructuredFluxDeltas, FaceIndexedFluxDeltas
export StructuredTransportWindow, FaceIndexedTransportWindow
export TransportBinaryDriver, driver_grid, air_mass_basis, load_transport_window
export has_humidity_endpoints, interpolate_fluxes!, expected_air_mass!, interpolate_qv!, copy_fluxes!
