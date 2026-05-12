# ---------------------------------------------------------------------------
# Forward-tape recording for the CS adjoint pipeline.
#
# Collects the per-op records the forward pass appends to the tape:
#
#   * `_CSTapeCounts` / `_tape_byte_estimate` — diagnostic sizing.
#   * `_record_sweep!` — pushes a `_CSSweepRecord` for one per-direction
#     advection sweep (linear or monotone-PPM dispatch).
#   * `_record_cs_mass_tape` — air-mass tape only (linear schemes).
#   * `_record_cs_tracer_tape` — air-mass + tracer-mass tape
#     (monotone PPM and other nonlinear schemes).
#   * `_record_cs_adjoint_tape` — top-level dispatch that picks the right
#     recorder for the requested scheme; LinRood goes through
#     `_record_cs_linrood_tape` in `LinRoodTape.jl`.
#
# Relocated from `src/Adjoints/Adjoints.jl` lines 586-702 and 718-1050
# unchanged in Plan 26 P0.3b; no semantic change. Loaded into the
# `Adjoints` module via an `include` from `Adjoints.jl`.
# ---------------------------------------------------------------------------

struct _CSTapeCounts
    sweep_records::Int
    halo_records::Int
    midpoint_records::Int
    diffusion_records::Int
    convection_records::Int
end

function _tape_byte_estimate(panels_m0,
                             panels_am_steps,
                             panels_bm_steps,
                             panels_cm_steps,
                             mesh::CubedSphereMesh,
                             scheme::CSAdjointSupportedScheme;
                             cfl_limit = 0.95,
                             diffusion_op = NoDiffusion(),
                             convection_op = NoConvection())
    FT = eltype(panels_m0[1])
    nsteps = _validate_step_sequences(panels_am_steps, panels_bm_steps,
                                      panels_cm_steps)
    panels_m = _copy_panel_tuple(panels_m0)
    fill_panel_halos!(panels_m, mesh; dir=0)
    dummy_rm = ntuple(p -> similar(panels_m[p]), 6)
    @inbounds for p in 1:6
        fill!(dummy_rm[p], zero(FT))
    end
    ws = CSAdvectionWorkspace(mesh, panels_m[1])
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(panels_m[1], 3)
    cfl_ft = FT(cfl_limit)

    sweep_records = 0
    halo_records = 0
    midpoint_records = 0
    diffusion_records = 0
    diffusion_state_records = 0
    convection_records = 0

    @inbounds for step in 1:nsteps
        panels_am = panels_am_steps[step]
        panels_bm = panels_bm_steps[step]
        panels_cm = panels_cm_steps[step]
        n_x = _cs_static_subcycle_count(panels_am, panels_m, Nc, Hp, Nz, cfl_ft, :x)
        n_y = _cs_static_subcycle_count(panels_bm, panels_m, Nc, Hp, Nz, cfl_ft, :y)
        n_z = _cs_static_subcycle_count(panels_cm, panels_m, Nc, Hp, Nz, cfl_ft, :z)

        sweep_records += 2n_x + 2n_y + 2n_z
        halo_records += 2n_x + 2n_y + 2
        midpoint_records += 1
        if !(_diffusion_sequence_at(diffusion_op, step, nsteps,
                                    "diffusion_op") isa NoDiffusion)
            diffusion_records += 2
            diffusion_state_records += 1
        end
        convection_op isa NoConvection || (convection_records += 1)

        fs_x = one(FT) / FT(n_x)
        fs_y = one(FT) / FT(n_y)
        fs_z = one(FT) / FT(n_z)

        for _ in 1:n_x
            for p in 1:6
                _sweep_x_panel!(dummy_rm[p], panels_m[p], panels_am[p], scheme,
                                ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_x)
            end
            fill_panel_halos!(panels_m, mesh; dir=1)
        end
        for _ in 1:n_y
            for p in 1:6
                _sweep_y_panel!(dummy_rm[p], panels_m[p], panels_bm[p], scheme,
                                ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_y)
            end
            fill_panel_halos!(panels_m, mesh; dir=2)
        end
        for _ in 1:n_z
            for p in 1:6
                _sweep_z_panel!(dummy_rm[p], panels_m[p], panels_cm[p], scheme,
                                ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_z)
            end
        end
        for _ in 1:n_z
            for p in 1:6
                _sweep_z_panel!(dummy_rm[p], panels_m[p], panels_cm[p], scheme,
                                ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_z)
            end
        end
        fill_panel_halos!(panels_m, mesh; dir=2)
        for _ in 1:n_y
            for p in 1:6
                _sweep_y_panel!(dummy_rm[p], panels_m[p], panels_bm[p], scheme,
                                ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_y)
            end
            fill_panel_halos!(panels_m, mesh; dir=2)
        end
        fill_panel_halos!(panels_m, mesh; dir=1)
        for _ in 1:n_x
            for p in 1:6
                _sweep_x_panel!(dummy_rm[p], panels_m[p], panels_am[p], scheme,
                                ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_x)
            end
            fill_panel_halos!(panels_m, mesh; dir=1)
        end
    end

    sweep_state_records = scheme isa CSAdjointNonlinearScheme ? 2sweep_records : sweep_records
    state_records = sweep_state_records + diffusion_state_records + convection_records
    total_records = state_records + halo_records + midpoint_records
    bytes_per_state = _bytes_per_panel_tuple(panels_m0)
    return CSTapeByteEstimate(nsteps, sweep_records, halo_records,
                              midpoint_records, diffusion_records,
                              convection_records, state_records,
                              total_records, bytes_per_state,
                              state_records * bytes_per_state)
end

cs_tape_byte_estimate(args...; kwargs...) =
    _tape_byte_estimate(args...; kwargs...)

# ---------------------------------------------------------------------------
# Per-step tape recorders
# ---------------------------------------------------------------------------

function _record_sweep!(ops, direction::Symbol, scheme::CSAdjointLinearScheme,
                        panels_m, panels_flux, flux_scale, tape_storage)
    push!(ops, _CSSweepRecord(direction, scheme,
                              _stage_panels(tape_storage, panels_m),
                              nothing,
                              panels_flux, flux_scale))
    return nothing
end

function _record_sweep!(ops, direction::Symbol, scheme::CSAdjointNonlinearScheme,
                        panels_m, panels_rm, panels_flux, flux_scale, tape_storage)
    push!(ops, _CSSweepRecord(direction, scheme,
                              _stage_panels(tape_storage, panels_m),
                              _stage_panels(tape_storage, panels_rm),
                              panels_flux, flux_scale))
    return nothing
end

function _record_cs_mass_tape(panels_m0,
                              panels_am_steps,
                              panels_bm_steps,
                              panels_cm_steps,
                              mesh::CubedSphereMesh,
                              scheme::CSAdjointLinearScheme;
                              cfl_limit,
                              flux_scale,
                              dt,
                              diffusion_op = NoDiffusion(),
                              diffusion_workspace = nothing,
                              convection_op = NoConvection(),
                              convection_forcing = nothing,
                              tape_storage = :device)
    FT = eltype(panels_m0[1])
    storage = _tape_storage(tape_storage)
    nsteps = _validate_step_sequences(panels_am_steps, panels_bm_steps, panels_cm_steps)
    panels_m = _copy_panel_tuple(panels_m0)
    fill_panel_halos!(panels_m, mesh; dir=0)
    dummy_rm = ntuple(p -> similar(panels_m[p]), 6)
    @inbounds for p in 1:6
        fill!(dummy_rm[p], zero(FT))
    end
    ws = CSAdvectionWorkspace(mesh, panels_m[1])
    ops = Any[]
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(panels_m[1], 3)
    cfl_ft = FT(cfl_limit)
    fs = FT(flux_scale)

    @inbounds for step in 1:nsteps
        panels_am = panels_am_steps[step]
        panels_bm = panels_bm_steps[step]
        panels_cm = panels_cm_steps[step]
        n_x = _cs_static_subcycle_count(panels_am, panels_m, Nc, Hp, Nz, cfl_ft, :x)
        n_y = _cs_static_subcycle_count(panels_bm, panels_m, Nc, Hp, Nz, cfl_ft, :y)
        n_z = _cs_static_subcycle_count(panels_cm, panels_m, Nc, Hp, Nz, cfl_ft, :z)
        fs_x = fs / FT(n_x)
        fs_y = fs / FT(n_y)
        fs_z = fs / FT(n_z)

        for _ in 1:n_x
            _record_sweep!(ops, :x, scheme, panels_m, panels_am, fs_x, storage)
            for p in 1:6
                _sweep_x_panel!(dummy_rm[p], panels_m[p], panels_am[p], scheme,
                                ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_x)
            end
            fill_panel_halos!(panels_m, mesh; dir=1)
            push!(ops, _CSHaloRecord(1))
        end

        for _ in 1:n_y
            _record_sweep!(ops, :y, scheme, panels_m, panels_bm, fs_y, storage)
            for p in 1:6
                _sweep_y_panel!(dummy_rm[p], panels_m[p], panels_bm[p], scheme,
                                ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_y)
            end
            fill_panel_halos!(panels_m, mesh; dir=2)
            push!(ops, _CSHaloRecord(2))
        end

        for _ in 1:n_z
            _record_sweep!(ops, :z, scheme, panels_m, panels_cm, fs_z, storage)
            for p in 1:6
                _sweep_z_panel!(dummy_rm[p], panels_m[p], panels_cm[p], scheme,
                                ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_z)
            end
        end

        diffusion_op_step = _diffusion_sequence_at(diffusion_op, step, nsteps,
                                                   "diffusion_op")
        if diffusion_op_step isa NoDiffusion
            push!(ops, _CSMidpointRecord(step))
        else
            diffusion_ws_step = _diffusion_sequence_at(diffusion_workspace, step,
                                                       nsteps,
                                                       "diffusion_workspace")
            panels_m_midpoint = _stage_panels(storage, panels_m)
            half_dt = FT(dt) / FT(2)
            push!(ops, _CSDiffusionRecord(diffusion_op_step, diffusion_ws_step,
                                          panels_m_midpoint, half_dt))
            push!(ops, _CSMidpointRecord(step))
            push!(ops, _CSDiffusionRecord(diffusion_op_step, diffusion_ws_step,
                                          panels_m_midpoint, half_dt))
        end

        for _ in 1:n_z
            _record_sweep!(ops, :z, scheme, panels_m, panels_cm, fs_z, storage)
            for p in 1:6
                _sweep_z_panel!(dummy_rm[p], panels_m[p], panels_cm[p], scheme,
                                ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_z)
            end
        end

        fill_panel_halos!(panels_m, mesh; dir=2)
        push!(ops, _CSHaloRecord(2))
        for _ in 1:n_y
            _record_sweep!(ops, :y, scheme, panels_m, panels_bm, fs_y, storage)
            for p in 1:6
                _sweep_y_panel!(dummy_rm[p], panels_m[p], panels_bm[p], scheme,
                                ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_y)
            end
            fill_panel_halos!(panels_m, mesh; dir=2)
            push!(ops, _CSHaloRecord(2))
        end

        fill_panel_halos!(panels_m, mesh; dir=1)
        push!(ops, _CSHaloRecord(1))
        for _ in 1:n_x
            _record_sweep!(ops, :x, scheme, panels_m, panels_am, fs_x, storage)
            for p in 1:6
                _sweep_x_panel!(dummy_rm[p], panels_m[p], panels_am[p], scheme,
                                ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_x)
            end
            fill_panel_halos!(panels_m, mesh; dir=1)
            push!(ops, _CSHaloRecord(1))
        end

        if !(convection_op isa NoConvection)
            forcing_step = _convection_forcing_at(convection_forcing, step, nsteps)
            forcing_step === nothing && throw(ArgumentError(
                "convection_op=$(typeof(convection_op)) requires `convection_forcing`"))
            push!(ops, _CSConvectionRecord(convection_op, forcing_step,
                                           _stage_panels(storage, panels_m),
                                           FT(dt)))
        end
    end

    return ops, panels_m
end

function _record_cs_tracer_tape(panels_rm0,
                                panels_m0,
                                panels_am_steps,
                                panels_bm_steps,
                                panels_cm_steps,
                                mesh::CubedSphereMesh,
                                scheme::CSAdjointNonlinearScheme;
                                cfl_limit,
                                flux_scale,
                                dt,
                                base_emission_rates = nothing,
                                diffusion_op = NoDiffusion(),
                                diffusion_workspace = nothing,
                                diffusion_meteo = nothing,
                                convection_op = NoConvection(),
                                convection_forcing = nothing,
                                convection_workspace = nothing,
                                tape_storage = :device)
    storage = _tape_storage(tape_storage)
    nsteps = _validate_step_sequences(panels_am_steps, panels_bm_steps, panels_cm_steps)
    _validate_emission_rates(base_emission_rates, nsteps, mesh, "base_emission_rates")
    panels_rm = _copy_panel_tuple(panels_rm0)
    panels_m = _copy_panel_tuple(panels_m0)
    fill_panel_halos!(panels_rm, mesh; dir=0)
    fill_panel_halos!(panels_m, mesh; dir=0)
    ws = CSAdvectionWorkspace(mesh, panels_m[1])
    ops = Any[]
    Nc, Hp = mesh.Nc, mesh.Hp
    Nz = size(panels_m[1], 3)
    FT = eltype(panels_m[1])
    cfl_ft = FT(cfl_limit)
    fs = FT(flux_scale)
    dt_ft = FT(dt)

    @inbounds for step in 1:nsteps
        panels_am = panels_am_steps[step]
        panels_bm = panels_bm_steps[step]
        panels_cm = panels_cm_steps[step]
        n_x = _cs_static_subcycle_count(panels_am, panels_m, Nc, Hp, Nz, cfl_ft, :x)
        n_y = _cs_static_subcycle_count(panels_bm, panels_m, Nc, Hp, Nz, cfl_ft, :y)
        n_z = _cs_static_subcycle_count(panels_cm, panels_m, Nc, Hp, Nz, cfl_ft, :z)
        fs_x = fs / FT(n_x)
        fs_y = fs / FT(n_y)
        fs_z = fs / FT(n_z)

        for _ in 1:n_x
            _record_sweep!(ops, :x, scheme, panels_m, panels_rm, panels_am, fs_x, storage)
            for p in 1:6
                _sweep_x_panel!(panels_rm[p], panels_m[p], panels_am[p],
                                scheme, ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_x)
            end
            fill_panel_halos!(panels_rm, mesh; dir=1)
            fill_panel_halos!(panels_m, mesh; dir=1)
            push!(ops, _CSHaloRecord(1))
        end

        for _ in 1:n_y
            _record_sweep!(ops, :y, scheme, panels_m, panels_rm, panels_bm, fs_y, storage)
            for p in 1:6
                _sweep_y_panel!(panels_rm[p], panels_m[p], panels_bm[p],
                                scheme, ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_y)
            end
            fill_panel_halos!(panels_rm, mesh; dir=2)
            fill_panel_halos!(panels_m, mesh; dir=2)
            push!(ops, _CSHaloRecord(2))
        end

        for _ in 1:n_z
            _record_sweep!(ops, :z, scheme, panels_m, panels_rm, panels_cm, fs_z, storage)
            for p in 1:6
                _sweep_z_panel!(panels_rm[p], panels_m[p], panels_cm[p],
                                scheme, ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_z)
            end
        end

        diffusion_op_step = _diffusion_sequence_at(diffusion_op, step, nsteps,
                                                   "diffusion_op")
        if diffusion_op_step isa NoDiffusion
            push!(ops, _CSMidpointRecord(step))
            base_emission_rates !== nothing &&
                _add_surface_rates!(panels_rm, base_emission_rates[step], dt_ft, mesh)
        else
            diffusion_ws_step = _diffusion_sequence_at(diffusion_workspace, step,
                                                       nsteps,
                                                       "diffusion_workspace")
            panels_m_midpoint = _stage_panels(storage, panels_m)
            half_dt = dt_ft / FT(2)
            push!(ops, _CSDiffusionRecord(diffusion_op_step, diffusion_ws_step,
                                          panels_m_midpoint, half_dt))
            apply_vertical_diffusion_vmr!(
                panels_rm, panels_m, diffusion_op_step, diffusion_ws_step,
                half_dt, diffusion_meteo; halo_width = mesh.Hp)
            push!(ops, _CSMidpointRecord(step))
            base_emission_rates !== nothing &&
                _add_surface_rates!(panels_rm, base_emission_rates[step], dt_ft, mesh)
            push!(ops, _CSDiffusionRecord(diffusion_op_step, diffusion_ws_step,
                                          panels_m_midpoint, half_dt))
            apply_vertical_diffusion_vmr!(
                panels_rm, panels_m, diffusion_op_step, diffusion_ws_step,
                half_dt, diffusion_meteo; halo_width = mesh.Hp)
        end

        for _ in 1:n_z
            _record_sweep!(ops, :z, scheme, panels_m, panels_rm, panels_cm, fs_z, storage)
            for p in 1:6
                _sweep_z_panel!(panels_rm[p], panels_m[p], panels_cm[p],
                                scheme, ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_z)
            end
        end

        fill_panel_halos!(panels_rm, mesh; dir=2)
        fill_panel_halos!(panels_m, mesh; dir=2)
        push!(ops, _CSHaloRecord(2))
        for _ in 1:n_y
            _record_sweep!(ops, :y, scheme, panels_m, panels_rm, panels_bm, fs_y, storage)
            for p in 1:6
                _sweep_y_panel!(panels_rm[p], panels_m[p], panels_bm[p],
                                scheme, ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_y)
            end
            fill_panel_halos!(panels_rm, mesh; dir=2)
            fill_panel_halos!(panels_m, mesh; dir=2)
            push!(ops, _CSHaloRecord(2))
        end

        fill_panel_halos!(panels_rm, mesh; dir=1)
        fill_panel_halos!(panels_m, mesh; dir=1)
        push!(ops, _CSHaloRecord(1))
        for _ in 1:n_x
            _record_sweep!(ops, :x, scheme, panels_m, panels_rm, panels_am, fs_x, storage)
            for p in 1:6
                _sweep_x_panel!(panels_rm[p], panels_m[p], panels_am[p],
                                scheme, ws.rm_A, ws.m_A, Nc, Hp, Nz; flux_scale=fs_x)
            end
            fill_panel_halos!(panels_rm, mesh; dir=1)
            fill_panel_halos!(panels_m, mesh; dir=1)
            push!(ops, _CSHaloRecord(1))
        end

        if !(convection_op isa NoConvection)
            forcing_step = _convection_forcing_at(convection_forcing, step, nsteps)
            forcing_step === nothing && throw(ArgumentError(
                "convection_op=$(typeof(convection_op)) requires `convection_forcing`"))
            push!(ops, _CSConvectionRecord(convection_op, forcing_step,
                                           _stage_panels(storage, panels_m),
                                           dt_ft))
            _apply_cs_convection_forward!(panels_rm, panels_m, forcing_step,
                                          convection_op, dt_ft,
                                          convection_workspace, mesh)
        end
    end

    return ops, panels_m
end

function _record_cs_adjoint_tape(panels_rm0, panels_m0,
                                 panels_am_steps, panels_bm_steps, panels_cm_steps,
                                 mesh::CubedSphereMesh,
                                 scheme::CSAdjointLinearScheme;
                                 base_emission_rates = nothing,
                                 diffusion_meteo = nothing,
                                 convection_workspace = nothing,
                                 kwargs...)
    return _record_cs_mass_tape(panels_m0,
                                panels_am_steps, panels_bm_steps, panels_cm_steps,
                                mesh, scheme; kwargs...)
end

function _record_cs_adjoint_tape(panels_rm0, panels_m0,
                                 panels_am_steps, panels_bm_steps, panels_cm_steps,
                                 mesh::CubedSphereMesh,
                                 scheme::CSAdjointLinRoodScheme; kwargs...)
    return _record_cs_linrood_tape(panels_rm0, panels_m0,
                                    panels_am_steps, panels_bm_steps,
                                    panels_cm_steps, mesh, scheme; kwargs...)
end

function _record_cs_adjoint_tape(panels_rm0, panels_m0,
                                 panels_am_steps, panels_bm_steps, panels_cm_steps,
                                 mesh::CubedSphereMesh,
                                 scheme::CSAdjointNonlinearScheme; kwargs...)
    return _record_cs_tracer_tape(panels_rm0, panels_m0,
                                  panels_am_steps, panels_bm_steps, panels_cm_steps,
                                  mesh, scheme; kwargs...)
end
