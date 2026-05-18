# ---------------------------------------------------------------------------
# Adjoint of CS implicit vertical diffusion.
#
# Reverse-mode of `ImplicitVerticalDiffusion`. Builds the same Thomas-
# solve coefficients as the forward pass on the fly, then runs the
# transpose solve (upper-triangular sweep first, then back-substitution).
# Mass-aware: enters and exits in lambda-on-tracer-mass space.
#
# Relocated from `src/Adjoints/Adjoints.jl` lines 1553-1722 unchanged
# in Plan 26 P0.2; no semantic change.
# ---------------------------------------------------------------------------

@inline function _adjoint_diffusion_time(::Type{FT}, meteo) where FT
    return meteo === nothing ? zero(FT) : FT(current_time(meteo))
end

@kernel function _vertical_diffusion_cs_single_adjoint_kernel!(
    lambda, @Const(air_mass), kz_field, @Const(dz), w_scratch,
    dt, Nz::Int, Hp::Int)
    ii, jj = @index(Global, NTuple)
    FT = eltype(lambda)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        dt_ft = FT(dt)

        w_prev = zero(FT)
        g_prev = zero(FT)

        for k in 1:Nz
            Kz_k = field_value(kz_field, (ii, jj, k))
            dz_k = dz[ii, jj, k]

            D_above = zero(FT)
            D_below = zero(FT)
            a_T = zero(FT)
            c_T = zero(FT)

            if k > 1
                Kz_prev = field_value(kz_field, (ii, jj, k - 1))
                dz_prev = dz[ii, jj, k - 1]
                Kz_above = (Kz_prev + Kz_k) / FT(2)
                dz_above = (dz_prev + dz_k) / FT(2)
                D_above = Kz_above / (dz_k * dz_above)
                a_T = -dt_ft * Kz_above / (dz_prev * dz_above)
            end

            if k < Nz
                Kz_next = field_value(kz_field, (ii, jj, k + 1))
                dz_next = dz[ii, jj, k + 1]
                Kz_below = (Kz_k + Kz_next) / FT(2)
                dz_below = (dz_k + dz_next) / FT(2)
                D_below = Kz_below / (dz_k * dz_below)
                c_T = -dt_ft * Kz_below / (dz_next * dz_below)
            end

            b_T = one(FT) + dt_ft * (D_above + D_below)
            m_k = air_mass[i, j, k]
            d_k = m_k > zero(FT) ? m_k * lambda[i, j, k] : zero(FT)

            if k == 1
                denom = b_T
                w_k = c_T / denom
                g_k = d_k / denom
            else
                denom = b_T - a_T * w_prev
                w_k = c_T / denom
                g_k = (d_k - a_T * g_prev) / denom
            end

            w_scratch[ii, jj, k] = w_k
            lambda[i, j, k] = g_k

            if k < Nz
                w_prev = w_k
                g_prev = g_k
            end
        end

        for k in (Nz - 1):-1:1
            lambda[i, j, k] = lambda[i, j, k] -
                              w_scratch[ii, jj, k] * lambda[i, j, k + 1]
        end

        for k in 1:Nz
            m_k = air_mass[i, j, k]
            lambda[i, j, k] = m_k > zero(FT) ? lambda[i, j, k] / m_k : zero(FT)
        end
    end
end

function _require_cs_diffusion_workspace(workspace)
    workspace === nothing && throw(ArgumentError(
        "CS adjoint diffusion requires a workspace with panel-native " *
        "`w_scratch` and `dz_scratch`; pass the transport CSAdvectionWorkspace"))
    hasproperty(workspace, :w_scratch) && hasproperty(workspace, :dz_scratch) ||
        throw(ArgumentError(
            "CS adjoint diffusion requires a workspace with panel-native " *
            "`w_scratch` and `dz_scratch` tuples"))
    w_scratch = getproperty(workspace, :w_scratch)
    dz_scratch = getproperty(workspace, :dz_scratch)
    length(w_scratch) == 6 && length(dz_scratch) == 6 ||
        throw(DimensionMismatch("CS adjoint diffusion workspace must provide 6 panel scratch arrays"))
    return w_scratch, dz_scratch
end

function _diffusion_sequence_at(value, step::Int, nsteps::Int,
                                name::AbstractString)
    if value isa AbstractVector
        length(value) == nsteps || throw(ArgumentError(
            "$name length $(length(value)) does not match nsteps $nsteps"))
        return value[step]
    else
        return value
    end
end

@inline _validate_cs_diffusion_kz_for_adjoint(_op) = nothing

function _validate_cs_diffusion_kz_for_adjoint(
    op::ImplicitVerticalDiffusion{FT, <:GCHPHoltslagBovilleKzField}) where FT
    @inbounds for p in 1:6
        data = panel_field(op.kz_field, p).data
        all(isfinite, data) || throw(ArgumentError(
            "GCHP VDIFF diffusion adjoint requires a finite Kz cache; " *
            "panel $p contains NaN or Inf. Refresh with " *
            "`refresh_gchp_holtslag_boville_kz_cache!` before recording the tape."))
        maximum(data) > zero(FT) || throw(ArgumentError(
            "GCHP VDIFF diffusion adjoint received an all-zero Kz cache on " *
            "panel $p. This usually means the VDIFF Kz field was constructed " *
            "but never refreshed from surface/vdiff forcing before the adjoint " *
            "tape was recorded."))
    end
    return nothing
end

function _validate_cs_diffusion_inputs(diffusion_op, diffusion_workspace,
                                       nsteps::Int)
    checked_kz = IdDict{Any, Bool}()
    for step in 1:nsteps
        op = _diffusion_sequence_at(diffusion_op, step, nsteps, "diffusion_op")
        if !(op isa NoDiffusion)
            if !haskey(checked_kz, op.kz_field)
                _validate_cs_diffusion_kz_for_adjoint(op)
                checked_kz[op.kz_field] = true
            end
            ws = _diffusion_sequence_at(diffusion_workspace, step, nsteps,
                                        "diffusion_workspace")
            _require_cs_diffusion_workspace(ws)
        end
    end
    return nothing
end

function _apply_cs_diffusion_adjoint!(lambda_panels, panels_m, ::NoDiffusion,
                                      workspace, dt, meteo,
                                      mesh::CubedSphereMesh)
    return nothing
end

function _apply_cs_diffusion_adjoint!(lambda_panels::NTuple{6, A},
                                      panels_m::NTuple{6},
                                      op::ImplicitVerticalDiffusion{FT, KzF},
                                      workspace, dt, meteo,
                                      mesh::CubedSphereMesh) where {
                                          FT, A <: AbstractArray{FT, 3},
                                          KzF <: AbstractCubedSphereField{FT}}
    w_scratch, dz_scratch = _require_cs_diffusion_workspace(workspace)
    update_field!(op.kz_field, _adjoint_diffusion_time(FT, meteo))

    Hp = mesh.Hp
    Nc = mesh.Nc
    @inbounds for p in 1:6
        panel_lambda = lambda_panels[p]
        panel_m = panels_m[p]
        size(panel_lambda) == size(panel_m) || throw(DimensionMismatch(
            "adjoint tracer panel $p shape $(size(panel_lambda)) does not match " *
            "air_mass shape $(size(panel_m))"))
        Nz = size(panel_lambda, 3)
        expected = (Nc, Nc, Nz)
        size(w_scratch[p]) == size(dz_scratch[p]) ||
            throw(DimensionMismatch("CS adjoint diffusion w_scratch and dz_scratch sizes differ on panel $p"))
        size(w_scratch[p]) == expected ||
            throw(DimensionMismatch(
                "CS adjoint diffusion workspace panel $p has shape $(size(w_scratch[p])); " *
                "expected $expected"))

        panel_kz = panel_field(op.kz_field, p)
        backend = get_backend(panel_lambda)
        kernel! = _vertical_diffusion_cs_single_adjoint_kernel!(backend, (8, 8))
        kernel!(panel_lambda, panel_m, panel_kz, dz_scratch[p], w_scratch[p],
                FT(dt), Nz, Hp; ndrange = (Nc, Nc))
        synchronize(backend)
    end
    return nothing
end

function _apply_cs_diffusion_adjoint!(lambda_panels, panels_m,
                                      op::ImplicitVerticalDiffusion,
                                      workspace, dt, meteo,
                                      mesh::CubedSphereMesh)
    throw(ArgumentError(
        "CS adjoint diffusion requires `ImplicitVerticalDiffusion` with a " *
        "`CubedSphereField` Kz field; got $(typeof(op.kz_field))"))
end
