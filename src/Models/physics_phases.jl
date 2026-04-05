# ---------------------------------------------------------------------------
# Physics phase functions — grid-dispatched wrappers for the unified run loop
#
# Each phase dispatches on grid type (LatitudeLongitudeGrid vs CubedSphereGrid).
# The unified run loop calls these in sequence without any if/else on grid type.
# ---------------------------------------------------------------------------

# =====================================================================
# Sub-step diagnostics for moist advection debugging
# =====================================================================

"""
Diagnostic container for capturing intermediate moist advection state.
Set `MOIST_DIAG[] = MoistSubStepDiag(...)` before `run!` to enable.
"""
mutable struct MoistSubStepDiag{FT}
    q_wet_post_hadv::Vector{Array{FT,3}}   # after gchp_tracer_2d!, substep 1
    rm_post_vremap::Vector{Array{FT,3}}    # after vertical_remap_cs!, substep 1
    qv_start::Vector{Array{FT,3}}          # QV at window start
    qv_back::Vector{Array{FT,3}}           # QV used for back-conversion
    delp_start::Vector{Array{FT,3}}        # moist DELP at start
    delp_end::Vector{Array{FT,3}}          # moist DELP at end
    q_dry_init::Vector{Array{FT,3}}        # dry MR before dry→wet conversion
    q_dry_final::Vector{Array{FT,3}}       # dry MR after wet→dry conversion
    captured::Bool                          # true once first window captured
end

function MoistSubStepDiag(FT::Type, Nc::Int, Nz::Int, Hp::Int)
    N = Nc + 2Hp
    alloc() = [zeros(FT, N, N, Nz) for _ in 1:6]
    MoistSubStepDiag{FT}(alloc(), alloc(), alloc(), alloc(),
                          alloc(), alloc(), alloc(), alloc(), false)
end

"""Global ref for moist sub-step diagnostics. Set to a `MoistSubStepDiag` to enable capture."""
const MOIST_DIAG = Ref{Any}(nothing)

"""
Diagnostic container for LL first-step transport audits.

Captures scalar stage metrics for the first few dynamic intervals and,
optionally, a sparse set of 2D snapshots for later inspection.
"""
mutable struct LLTransportAudit
    tracer::Symbol
    max_windows::Int
    max_dyn_intervals::Int
    capture_snapshots::Bool
    include_tm5_reference::Bool
    sh_ocean_mask::Union{Nothing, BitMatrix}
    zonal_lat_index::Int
    lon_values::Vector{Float64}
    lat_values::Vector{Float64}
    qv_start
    qv_next
    threshold_surface_ppm::Float64
    threshold_column_ppm::Float64
    stages::Vector{Any}
    snapshots::Dict{Int, Any}
    window_baselines::Dict{Int, Any}
    metadata::Dict{String, Any}
end

function LLTransportAudit(; tracer::Symbol=:co2,
                          max_windows::Int=2,
                          max_dyn_intervals::Int=2,
                          capture_snapshots::Bool=true,
                          include_tm5_reference::Bool=true,
                          sh_ocean_mask::Union{Nothing, BitMatrix}=nothing,
                          zonal_lat_index::Int=0,
                          lon_values::AbstractVector{<:Real}=Float64[],
                          lat_values::AbstractVector{<:Real}=Float64[],
                          qv_start=nothing,
                          qv_next=nothing,
                          threshold_surface_ppm::Real=0.1,
                          threshold_column_ppm::Real=0.5)
    return LLTransportAudit(
        tracer,
        max_windows,
        max_dyn_intervals,
        capture_snapshots,
        include_tm5_reference,
        sh_ocean_mask,
        zonal_lat_index,
        Float64.(lon_values),
        Float64.(lat_values),
        qv_start,
        qv_next,
        Float64(threshold_surface_ppm),
        Float64(threshold_column_ppm),
        Any[],
        Dict{Int, Any}(),
        Dict{Int, Any}(),
        Dict{String, Any}(),
    )
end

"""Global ref for LL transport audit capture. Set to `LLTransportAudit(...)` to enable."""
const LL_AUDIT = Ref{Any}(nothing)

_ll_advection_cfg(model) = get(get(model.metadata, "config", Dict()), "advection", Dict())

function _ll_debug_first_window(model)
    adv_cfg = _ll_advection_cfg(model)
    return get(adv_cfg, "debug_first_window", false) ||
           get(adv_cfg, "debug_ll_first_window", false)
end

function _ll_debug_window(model)
    adv_cfg = _ll_advection_cfg(model)
    win = get(adv_cfg, "debug_window", nothing)
    return win === nothing ? nothing : Int(win)
end

function _ll_debug_all_substeps(model)
    adv_cfg = _ll_advection_cfg(model)
    return get(adv_cfg, "debug_all_substeps", false)
end

function _ll_use_flux_delta(model, gpu)
    adv_cfg = _ll_advection_cfg(model)
    enabled = get(adv_cfg, "enable_flux_delta", get(adv_cfg, "use_flux_delta", false))
    enabled || return false
    gpu.dam === nothing && return false
    gpu.dbm === nothing && return false
    gpu.dm === nothing && return false
    return length(gpu.dam) > 0 && length(gpu.dbm) > 0 && length(gpu.dm) > 0
end

function _ll_use_mass_cfl_pilot(model)
    adv_cfg = _ll_advection_cfg(model)
    return get(adv_cfg, "use_mass_cfl_pilot",
               get(adv_cfg, "mass_cfl_pilot", true))
end

function _ll_mass_cfl_pilot_max_refinement(model)
    adv_cfg = _ll_advection_cfg(model)
    return Int(get(adv_cfg, "mass_cfl_pilot_max_refinement",
                   get(adv_cfg, "mass_cfl_max_refinement", 64)))
end

_ll_massflux_workspace(adv_ws) = hasproperty(adv_ws, :base) ? adv_ws.base : adv_ws

function _finite_array_stats(arr)
    host = Array(arr)
    n_nan = 0
    n_inf = 0
    n_nonpos = 0
    n_finite = 0
    minv = 0.0
    maxv = 0.0
    have_finite = false

    @inbounds for x in host
        if isnan(x)
            n_nan += 1
        elseif isinf(x)
            n_inf += 1
        else
            xf = Float64(x)
            n_finite += 1
            x <= zero(x) && (n_nonpos += 1)
            if have_finite
                minv = min(minv, xf)
                maxv = max(maxv, xf)
            else
                minv = xf
                maxv = xf
                have_finite = true
            end
        end
    end

    return (; n_nan, n_inf, n_nonpos, n_finite,
             min=have_finite ? minv : NaN,
             max=have_finite ? maxv : NaN)
end

function _finite_ratio_stats(rm, m)
    n_nan = 0
    n_inf = 0
    n_nonpos = 0
    n_finite = 0
    minv = 0.0
    maxv = 0.0
    have_finite = false

    @inbounds for idx in eachindex(rm, m)
        rmv = Float64(rm[idx])
        mv = Float64(m[idx])
        cv = if isfinite(rmv) && isfinite(mv) && mv != 0.0
            rmv / mv
        elseif isfinite(rmv) && mv == 0.0
            rmv >= 0.0 ? Inf : -Inf
        else
            NaN
        end

        if isnan(cv)
            n_nan += 1
        elseif isinf(cv)
            n_inf += 1
        else
            n_finite += 1
            cv <= 0.0 && (n_nonpos += 1)
            if have_finite
                minv = min(minv, cv)
                maxv = max(maxv, cv)
            else
                minv = cv
                maxv = cv
                have_finite = true
            end
        end
    end

    return (; n_nan, n_inf, n_nonpos, n_finite,
             min=have_finite ? minv : NaN,
             max=have_finite ? maxv : NaN)
end

function _ll_surface_column_fields(rm::Array{FT,3}, m::Array{FT,3}) where FT
    Nx, Ny, Nz = size(rm)
    surface = Array{Float32}(undef, Nx, Ny)
    column = Array{Float32}(undef, Nx, Ny)
    surface_rm = Array{Float32}(undef, Nx, Ny)
    surface_m = Array{Float32}(undef, Nx, Ny)
    column_rm = Array{Float32}(undef, Nx, Ny)
    column_m = Array{Float32}(undef, Nx, Ny)

    @inbounds for j in 1:Ny, i in 1:Nx
        ms = Float64(m[i, j, Nz])
        rs = Float64(rm[i, j, Nz])
        surface_rm[i, j] = Float32(rs)
        surface_m[i, j] = Float32(ms)
        surface[i, j] = isfinite(ms) && isfinite(rs) && ms != 0.0 ?
                        Float32(1.0e6 * (rs / ms)) : Float32(NaN)

        sum_rm = 0.0
        sum_m = 0.0
        valid = true
        for k in 1:Nz
            rv = Float64(rm[i, j, k])
            mv = Float64(m[i, j, k])
            if !isfinite(rv) || !isfinite(mv)
                valid = false
                break
            end
            sum_rm += rv
            sum_m += mv
        end
        column_rm[i, j] = Float32(valid ? sum_rm : NaN)
        column_m[i, j] = Float32(valid ? sum_m : NaN)
        column[i, j] = valid && sum_m != 0.0 ? Float32(1.0e6 * (sum_rm / sum_m)) : Float32(NaN)
    end

    return (; surface_ppm=surface, column_ppm=column,
             surface_rm, surface_m, column_rm, column_m)
end

function _ll_surface_column_ppm(rm::Array{FT,3}, m::Array{FT,3}) where FT
    fields = _ll_surface_column_fields(rm, m)
    return fields.surface_ppm, fields.column_ppm
end

function _ll_dry_mass_array(m::Array{FT,3}, qv::Array) where FT
    @assert size(m) == size(qv)
    m_dry = Array{FT}(undef, size(m))
    @inbounds for idx in eachindex(m, qv)
        m_dry[idx] = m[idx] * (one(FT) - FT(qv[idx]))
    end
    return m_dry
end

function _mean_std_masked(field::AbstractMatrix, mask::BitMatrix)
    @assert size(field) == size(mask)
    n = 0
    s = 0.0
    @inbounds for j in axes(field, 2), i in axes(field, 1)
        mask[i, j] || continue
        v = Float64(field[i, j])
        isfinite(v) || continue
        s += v
        n += 1
    end
    n == 0 && return (mean=NaN, std=NaN)
    meanv = s / n
    ss = 0.0
    @inbounds for j in axes(field, 2), i in axes(field, 1)
        mask[i, j] || continue
        v = Float64(field[i, j])
        isfinite(v) || continue
        dv = v - meanv
        ss += dv * dv
    end
    return (mean=meanv, std=sqrt(ss / n))
end

function _mean_std_row(field::AbstractMatrix, row::Int)
    row < first(axes(field, 2)) && return (mean=NaN, std=NaN)
    row > last(axes(field, 2)) && return (mean=NaN, std=NaN)
    n = 0
    s = 0.0
    @inbounds for i in axes(field, 1)
        v = Float64(field[i, row])
        isfinite(v) || continue
        s += v
        n += 1
    end
    n == 0 && return (mean=NaN, std=NaN)
    meanv = s / n
    ss = 0.0
    @inbounds for i in axes(field, 1)
        v = Float64(field[i, row])
        isfinite(v) || continue
        dv = v - meanv
        ss += dv * dv
    end
    return (mean=meanv, std=sqrt(ss / n))
end

function _corr_masked(x::AbstractMatrix, y::AbstractMatrix, mask::BitMatrix)
    @assert size(x) == size(y) == size(mask)
    n = 0
    sx = 0.0
    sy = 0.0
    @inbounds for j in axes(x, 2), i in axes(x, 1)
        mask[i, j] || continue
        xv = Float64(x[i, j])
        yv = Float64(y[i, j])
        (isfinite(xv) && isfinite(yv)) || continue
        sx += xv
        sy += yv
        n += 1
    end
    n < 2 && return NaN
    mx = sx / n
    my = sy / n
    sxx = 0.0
    syy = 0.0
    sxy = 0.0
    @inbounds for j in axes(x, 2), i in axes(x, 1)
        mask[i, j] || continue
        xv = Float64(x[i, j])
        yv = Float64(y[i, j])
        (isfinite(xv) && isfinite(yv)) || continue
        dx = xv - mx
        dy = yv - my
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    end
    (sxx > 0.0 && syy > 0.0) || return NaN
    return sxy / sqrt(sxx * syy)
end

function _ratio_delta_components(num1::AbstractMatrix, den1::AbstractMatrix,
                                 num0::AbstractMatrix, den0::AbstractMatrix)
    total = Array{Float32}(undef, size(num1))
    delta_num = Array{Float32}(undef, size(num1))
    delta_mass = Array{Float32}(undef, size(num1))
    rel_mass = Array{Float32}(undef, size(num1))
    @inbounds for j in axes(num1, 2), i in axes(num1, 1)
        n1 = Float64(num1[i, j]); d1 = Float64(den1[i, j])
        n0 = Float64(num0[i, j]); d0 = Float64(den0[i, j])
        if isfinite(n1) && isfinite(d1) && isfinite(n0) && isfinite(d0) && d1 != 0.0 && d0 != 0.0
            c1 = n1 / d1
            c0 = n0 / d0
            total[i, j] = Float32(1.0e6 * (c1 - c0))
            delta_num[i, j] = Float32(1.0e6 * ((n1 - n0) / d0))
            delta_mass[i, j] = Float32(1.0e6 * (n1 * (1.0 / d1 - 1.0 / d0)))
            rel_mass[i, j] = Float32((d1 - d0) / d0)
        else
            total[i, j] = Float32(NaN)
            delta_num[i, j] = Float32(NaN)
            delta_mass[i, j] = Float32(NaN)
            rel_mass[i, j] = Float32(NaN)
        end
    end
    return (; total, delta_num, delta_mass, rel_mass)
end

function _delta_component_metrics(comps, mask::Union{Nothing, BitMatrix}, row::Int)
    total_sh = mask === nothing ? (mean=NaN, std=NaN) : _mean_std_masked(comps.total, mask)
    num_sh = mask === nothing ? (mean=NaN, std=NaN) : _mean_std_masked(comps.delta_num, mask)
    mass_sh = mask === nothing ? (mean=NaN, std=NaN) : _mean_std_masked(comps.delta_mass, mask)
    rel_mass_sh = mask === nothing ? (mean=NaN, std=NaN) : _mean_std_masked(comps.rel_mass, mask)
    total_row = row > 0 ? _mean_std_row(comps.total, row) : (mean=NaN, std=NaN)
    num_row = row > 0 ? _mean_std_row(comps.delta_num, row) : (mean=NaN, std=NaN)
    mass_row = row > 0 ? _mean_std_row(comps.delta_mass, row) : (mean=NaN, std=NaN)
    rel_mass_corr = mask === nothing ? NaN : _corr_masked(comps.total, comps.rel_mass, mask)
    mass_corr = mask === nothing ? NaN : _corr_masked(comps.total, comps.delta_mass, mask)
    num_corr = mask === nothing ? NaN : _corr_masked(comps.total, comps.delta_num, mask)
    return (
        total_sh_std = Float64(total_sh.std),
        num_sh_std = Float64(num_sh.std),
        mass_sh_std = Float64(mass_sh.std),
        rel_mass_sh_std = Float64(rel_mass_sh.std),
        total_row_std = Float64(total_row.std),
        num_row_std = Float64(num_row.std),
        mass_row_std = Float64(mass_row.std),
        total_vs_rel_mass_corr = Float64(rel_mass_corr),
        total_vs_mass_corr = Float64(mass_corr),
        total_vs_num_corr = Float64(num_corr),
    )
end

function _surface_hotspot(surface_ppm::AbstractMatrix,
                          lon_values::AbstractVector{<:Real},
                          lat_values::AbstractVector{<:Real})
    meanv = 0.0
    n = 0
    @inbounds for v in surface_ppm
        vf = Float64(v)
        isfinite(vf) || continue
        meanv += vf
        n += 1
    end
    n == 0 && return (lon=NaN, lat=NaN, value=NaN, deviation=NaN)
    meanv /= n

    best_dev = -1.0
    best_i = 1
    best_j = 1
    best_v = NaN
    @inbounds for j in axes(surface_ppm, 2), i in axes(surface_ppm, 1)
        v = Float64(surface_ppm[i, j])
        isfinite(v) || continue
        dev = abs(v - meanv)
        if dev > best_dev
            best_dev = dev
            best_i = i
            best_j = j
            best_v = v
        end
    end

    lon = isempty(lon_values) ? NaN : Float64(lon_values[best_i])
    lat = isempty(lat_values) ? NaN : Float64(lat_values[best_j])
    return (lon=lon, lat=lat, value=best_v, deviation=best_dev)
end

function _ll_tm5_massflow_reference(am::AbstractArray{FT,3},
                                    bm::AbstractArray{FT,3},
                                    grid::LatitudeLongitudeGrid) where FT
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    pit = zeros(FT, Nx, Ny)
    sd = Nz > 1 ? zeros(FT, Nx, Ny, Nz - 1) : zeros(FT, Nx, Ny, 0)
    cm = zeros(FT, Nx, Ny, Nz + 1)
    B_ifc = hasproperty(grid.vertical, :B) ? Float64.(grid.vertical.B) : Float64[]
    bt = (!isempty(B_ifc) && length(B_ifc) == Nz + 1) ?
         Float64.(B_ifc[2:end] .- B_ifc[1:end-1]) :
         fill(1.0 / max(Nz, 1), Nz)

    @inbounds for j in 1:Ny, i in 1:Nx
        for k in 1:Nz
            pit[i, j] += am[i, j, k] - am[i + 1, j, k] + bm[i, j, k] - bm[i, j + 1, k]
        end
        if Nz > 1
            sd_ij = am[i, j, Nz] - am[i + 1, j, Nz] + bm[i, j, Nz] - bm[i, j + 1, Nz]
            sd_ij -= FT(bt[Nz] * Float64(pit[i, j]))
            sd[i, j, Nz - 1] = sd_ij
            for k in Nz - 2:-1:1
                conv_next = am[i, j, k + 1] - am[i + 1, j, k + 1] +
                            bm[i, j, k + 1] - bm[i, j + 1, k + 1]
                sd_ij = sd_ij + conv_next - FT(bt[k + 1] * Float64(pit[i, j]))
                sd[i, j, k] = sd_ij
            end
            for k in 1:Nz-1
                cm[i, j, k + 1] = -sd[i, j, k]
            end
        end
    end

    return (; pit, sd, cm)
end

function _ll_tm5_z_gamma_max(cm::AbstractArray{FT,3}, m::AbstractArray{FT,3}) where FT
    Nx, Ny, Nz = size(m)
    gamma_max = 0.0
    @inbounds for k in 1:Nz+1, j in 1:Ny, i in 1:Nx
        cmv = Float64(cm[i, j, k])
        cmv == 0.0 && continue
        donor = if cmv > 0.0
            (k > 1 && k <= Nz + 1) ? Float64(m[i, j, k - 1]) : 0.0
        else
            k <= Nz ? Float64(m[i, j, k]) : 0.0
        end
        donor > 0.0 || continue
        gamma_max = max(gamma_max, abs(cmv) / donor)
    end
    return gamma_max
end

function _ll_tm5_advectx_get_nloop_row(m_row::AbstractVector{FT},
                                       am_row::AbstractVector{FT};
                                       is_cyclic::Bool=true,
                                       max_nloop::Int=50) where FT
    Nx = length(m_row)
    length(am_row) == Nx + 1 || error("Expected Nx+1 x-faces, got $(length(am_row)) for Nx=$Nx")
    am_work = collect(am_row)
    nloop = 1
    alpha = 2.0

    while abs(alpha) >= 1.0 && nloop < max_nloop
        mx = zeros(Float64, Nx + 2)
        @inbounds for i in 1:Nx
            mx[i + 1] = Float64(m_row[i])
        end
        if is_cyclic
            mx[1] = mx[Nx + 1]
            mx[Nx + 2] = mx[2]
        else
            mx[1] = mx[2]
            mx[Nx + 2] = mx[Nx + 1]
        end

        cfl_ok = false
        alpha = 0.0
        for _iloop in 1:nloop
            cfl_ok = true
            for f in 1:Nx+1
                left = f == 1 ? Nx : f - 1
                right = f == Nx + 1 ? 1 : f
                a = Float64(am_work[f])
                donor = a >= 0.0 ? mx[left + 1] : mx[right + 1]
                local_alpha = donor != 0.0 ? a / donor : Inf
                alpha = max(alpha, abs(local_alpha))
                if abs(local_alpha) >= 1.0
                    cfl_ok = false
                    break
                end
            end
            cfl_ok || break
            for i in 1:Nx
                mx[i + 1] = mx[i + 1] + Float64(am_work[i]) - Float64(am_work[i + 1])
            end
            if is_cyclic
                mx[1] = mx[Nx + 1]
                mx[Nx + 2] = mx[2]
            end
        end

        if !cfl_ok
            am_work .*= FT(nloop) / FT(nloop + 1)
            nloop += 1
        else
            break
        end
    end

    return nloop
end

function _ll_tm5_x_nloop_max(m::AbstractArray{FT,3},
                             am::AbstractArray{FT,3},
                             cluster_sizes::AbstractVector) where FT
    Nx, Ny, Nz = size(m)
    cs_host = cluster_sizes isa Array ? cluster_sizes : Array(cluster_sizes)
    nloop_max = 1
    @inbounds for k in 1:Nz, j in 1:Ny
        r = Int(cs_host[j])
        if r <= 1
            row = @view m[:, j, k]
            faces = @view am[:, j, k]
            nloop_max = max(nloop_max, _ll_tm5_advectx_get_nloop_row(row, faces))
        else
            Nx_red = Nx ÷ r
            row_red = Array{FT}(undef, Nx_red)
            faces_red = Array{FT}(undef, Nx_red + 1)
            for ic in 1:Nx_red
                s = zero(FT)
                for off in 0:r-1
                    s += m[(ic - 1) * r + 1 + off, j, k]
                end
                row_red[ic] = s
                faces_red[ic] = am[(ic - 1) * r + 1, j, k]
            end
            faces_red[end] = am[Nx + 1, j, k]
            nloop_max = max(nloop_max, _ll_tm5_advectx_get_nloop_row(row_red, faces_red))
        end
    end
    return nloop_max
end

function _ll_tm5_reference_summary(am::AbstractArray{FT,3},
                                   bm::AbstractArray{FT,3},
                                   cm::AbstractArray{FT,3},
                                   m::AbstractArray{FT,3},
                                   grid::LatitudeLongitudeGrid,
                                   cluster_sizes::AbstractVector) where FT
    ref = _ll_tm5_massflow_reference(am, bm, grid)
    diff_absmax = 0.0
    diff_ss = 0.0
    n = 0
    @inbounds for idx in eachindex(cm, ref.cm)
        dv = Float64(cm[idx]) - Float64(ref.cm[idx])
        diff_absmax = max(diff_absmax, abs(dv))
        diff_ss += dv * dv
        n += 1
    end
    pit_absmax = 0.0
    @inbounds for v in ref.pit
        pit_absmax = max(pit_absmax, abs(Float64(v)))
    end
    sd_absmax = 0.0
    @inbounds for v in ref.sd
        sd_absmax = max(sd_absmax, abs(Float64(v)))
    end
    return (
        cm_absmax_diff = diff_absmax,
        cm_rms_diff = n > 0 ? sqrt(diff_ss / n) : NaN,
        gamma_max = _ll_tm5_z_gamma_max(cm, m),
        gamma_ref_max = _ll_tm5_z_gamma_max(ref.cm, m),
        x_nloop_max = _ll_tm5_x_nloop_max(m, am, cluster_sizes),
        pit_absmax = pit_absmax,
        sd_absmax = sd_absmax,
    )
end

function _ll_audit_enabled(audit, current_window::Int, dyn_interval::Int)
    audit isa LLTransportAudit || return false
    current_window <= audit.max_windows || return false
    dyn_interval <= audit.max_dyn_intervals || return false
    return true
end

function _ll_audit_should_snapshot(audit::LLTransportAudit,
                                   stage::AbstractString,
                                   micro::Int,
                                   refinement::Int,
                                   m_stats,
                                   rm_stats,
                                   c_stats)
    audit.capture_snapshots || return false
    stage == "window_start" && return true
    stage == "substep_setup" && return true
    micro == 1 && return true
    micro == refinement && return true
    m_stats.n_nan > 0 && return true
    rm_stats.n_nan > 0 && return true
    c_stats.n_nan > 0 && return true
    m_stats.n_nonpos > 0 && return true
    rm_stats.n_nonpos > 0 && return true
    return false
end

function _ll_audit_stage!(audit::LLTransportAudit,
                          stage::AbstractString,
                          grid::LatitudeLongitudeGrid,
                          rm_state,
                          m_state;
                          window::Int,
                          dyn_interval::Int,
                          substep::Int,
                          micro::Int,
                          refinement::Int,
                          time_fraction::Real=NaN,
                          qv_loaded::Bool=false,
                          has_deltas::Bool=false,
                          am=nothing,
                          bm=nothing,
                          cm=nothing,
                          cm_preclamp=nothing,
                          m0=nothing,
                          dm=nothing,
                          cluster_sizes=nothing,
                          qv=nothing)
    _ll_audit_enabled(audit, window, dyn_interval) || return nothing

    rm_arr = rm_state isa NamedTuple ? Array(rm_state[audit.tracer]) : Array(rm_state)
    m_arr = Array(m_state)
    qv_arr = if audit.qv_start !== nothing
        if audit.qv_next !== nothing && isfinite(Float64(time_fraction))
            FTq = eltype(audit.qv_start)
            audit.qv_start .+ FTq(time_fraction) .* (audit.qv_next .- audit.qv_start)
        else
            audit.qv_start
        end
    elseif qv === nothing
        nothing
    else
        qv_host = Array(qv)
        size(qv_host) == size(m_arr) ? qv_host : nothing
    end
    m_dry_arr = qv_arr === nothing ? nothing : _ll_dry_mass_array(m_arr, qv_arr)

    m_stats = _finite_array_stats(m_arr)
    rm_stats = _finite_array_stats(rm_arr)
    c_stats = _finite_ratio_stats(rm_arr, m_arr)
    diag_fields = _ll_surface_column_fields(rm_arr, m_arr)
    c_dry_stats = m_dry_arr === nothing ? nothing : _finite_ratio_stats(rm_arr, m_dry_arr)
    dry_diag_fields = m_dry_arr === nothing ? nothing : _ll_surface_column_fields(rm_arr, m_dry_arr)
    surface_ppm = diag_fields.surface_ppm
    column_ppm = diag_fields.column_ppm

    if stage == "window_start" && !haskey(audit.window_baselines, window)
        audit.window_baselines[window] = (
            surface_rm = copy(diag_fields.surface_rm),
            surface_m = copy(diag_fields.surface_m),
            column_rm = copy(diag_fields.column_rm),
            column_m = copy(diag_fields.column_m),
            dry_surface_rm = dry_diag_fields === nothing ? nothing : copy(dry_diag_fields.surface_rm),
            dry_surface_m = dry_diag_fields === nothing ? nothing : copy(dry_diag_fields.surface_m),
            dry_column_rm = dry_diag_fields === nothing ? nothing : copy(dry_diag_fields.column_rm),
            dry_column_m = dry_diag_fields === nothing ? nothing : copy(dry_diag_fields.column_m),
        )
    end
    baseline = get(audit.window_baselines, window, nothing)

    sh_surface = audit.sh_ocean_mask === nothing ? (mean=NaN, std=NaN) :
                 _mean_std_masked(surface_ppm, audit.sh_ocean_mask)
    sh_column = audit.sh_ocean_mask === nothing ? (mean=NaN, std=NaN) :
                _mean_std_masked(column_ppm, audit.sh_ocean_mask)
    zonal_surface = audit.zonal_lat_index > 0 ? _mean_std_row(surface_ppm, audit.zonal_lat_index) :
                    (mean=NaN, std=NaN)
    zonal_column = audit.zonal_lat_index > 0 ? _mean_std_row(column_ppm, audit.zonal_lat_index) :
                   (mean=NaN, std=NaN)
    dry_sh_surface = dry_diag_fields === nothing ? (mean=NaN, std=NaN) :
                     (audit.sh_ocean_mask === nothing ? (mean=NaN, std=NaN) :
                      _mean_std_masked(dry_diag_fields.surface_ppm, audit.sh_ocean_mask))
    dry_sh_column = dry_diag_fields === nothing ? (mean=NaN, std=NaN) :
                    (audit.sh_ocean_mask === nothing ? (mean=NaN, std=NaN) :
                     _mean_std_masked(dry_diag_fields.column_ppm, audit.sh_ocean_mask))
    dry_zonal_surface = dry_diag_fields === nothing || audit.zonal_lat_index <= 0 ? (mean=NaN, std=NaN) :
                        _mean_std_row(dry_diag_fields.surface_ppm, audit.zonal_lat_index)
    dry_zonal_column = dry_diag_fields === nothing || audit.zonal_lat_index <= 0 ? (mean=NaN, std=NaN) :
                       _mean_std_row(dry_diag_fields.column_ppm, audit.zonal_lat_index)
    hotspot = _surface_hotspot(surface_ppm, audit.lon_values, audit.lat_values)

    surface_delta = baseline === nothing ? nothing :
        _ratio_delta_components(diag_fields.surface_rm, diag_fields.surface_m,
                                baseline.surface_rm, baseline.surface_m)
    column_delta = baseline === nothing ? nothing :
        _ratio_delta_components(diag_fields.column_rm, diag_fields.column_m,
                                baseline.column_rm, baseline.column_m)
    surface_delta_metrics = surface_delta === nothing ? nothing :
        _delta_component_metrics(surface_delta, audit.sh_ocean_mask, audit.zonal_lat_index)
    column_delta_metrics = column_delta === nothing ? nothing :
        _delta_component_metrics(column_delta, audit.sh_ocean_mask, audit.zonal_lat_index)
    dry_surface_delta = baseline === nothing || dry_diag_fields === nothing || baseline.dry_surface_rm === nothing ? nothing :
        _ratio_delta_components(dry_diag_fields.surface_rm, dry_diag_fields.surface_m,
                                baseline.dry_surface_rm, baseline.dry_surface_m)
    dry_column_delta = baseline === nothing || dry_diag_fields === nothing || baseline.dry_column_rm === nothing ? nothing :
        _ratio_delta_components(dry_diag_fields.column_rm, dry_diag_fields.column_m,
                                baseline.dry_column_rm, baseline.dry_column_m)
    dry_surface_delta_metrics = dry_surface_delta === nothing ? nothing :
        _delta_component_metrics(dry_surface_delta, audit.sh_ocean_mask, audit.zonal_lat_index)
    dry_column_delta_metrics = dry_column_delta === nothing ? nothing :
        _delta_component_metrics(dry_column_delta, audit.sh_ocean_mask, audit.zonal_lat_index)

    am_absmax = am === nothing ? NaN : maximum(abs, Array(am))
    bm_absmax = bm === nothing ? NaN : maximum(abs, Array(bm))
    cm_absmax = cm === nothing ? NaN : maximum(abs, Array(cm))
    cm_pre_absmax = cm_preclamp === nothing ? NaN : maximum(abs, Array(cm_preclamp))
    cm_post_gamma_max = cm === nothing ? NaN : _ll_tm5_z_gamma_max(Array(cm), m_arr)
    cm_pre_gamma_max = cm_preclamp === nothing ? NaN : _ll_tm5_z_gamma_max(Array(cm_preclamp), m_arr)
    m0_stats = m0 === nothing ? nothing : _finite_array_stats(Array(m0))
    dm_stats = dm === nothing ? nothing : _finite_array_stats(Array(dm))

    tm5_ref = if audit.include_tm5_reference &&
                 am !== nothing && bm !== nothing && cm !== nothing && cluster_sizes !== nothing
        _ll_tm5_reference_summary(Array(am), Array(bm), Array(cm), m_arr, grid, cluster_sizes)
    else
        nothing
    end

    record = (
        label = String(stage),
        window = window,
        dyn_interval = dyn_interval,
        substep = substep,
        micro = micro,
        refinement = refinement,
        time_fraction = Float64(time_fraction),
        qv_loaded = qv_loaded,
        has_deltas = has_deltas,
        m_min = Float64(m_stats.min),
        m_max = Float64(m_stats.max),
        m_n_nan = Int(m_stats.n_nan),
        m_n_inf = Int(m_stats.n_inf),
        m_n_nonpos = Int(m_stats.n_nonpos),
        rm_min = Float64(rm_stats.min),
        rm_max = Float64(rm_stats.max),
        rm_n_nan = Int(rm_stats.n_nan),
        rm_n_inf = Int(rm_stats.n_inf),
        rm_n_nonpos = Int(rm_stats.n_nonpos),
        c_min_ppm = Float64(c_stats.min) * 1.0e6,
        c_max_ppm = Float64(c_stats.max) * 1.0e6,
        c_n_nan = Int(c_stats.n_nan),
        c_n_inf = Int(c_stats.n_inf),
        c_n_nonpos = Int(c_stats.n_nonpos),
        c_dry_min_ppm = c_dry_stats === nothing ? NaN : Float64(c_dry_stats.min) * 1.0e6,
        c_dry_max_ppm = c_dry_stats === nothing ? NaN : Float64(c_dry_stats.max) * 1.0e6,
        c_dry_n_nan = c_dry_stats === nothing ? -1 : Int(c_dry_stats.n_nan),
        c_dry_n_inf = c_dry_stats === nothing ? -1 : Int(c_dry_stats.n_inf),
        c_dry_n_nonpos = c_dry_stats === nothing ? -1 : Int(c_dry_stats.n_nonpos),
        surface_sh_ocean_mean_ppm = Float64(sh_surface.mean),
        surface_sh_ocean_std_ppm = Float64(sh_surface.std),
        surface_zonal50_std_ppm = Float64(zonal_surface.std),
        column_sh_ocean_mean_ppm = Float64(sh_column.mean),
        column_sh_ocean_std_ppm = Float64(sh_column.std),
        column_zonal50_std_ppm = Float64(zonal_column.std),
        dry_surface_sh_ocean_mean_ppm = Float64(dry_sh_surface.mean),
        dry_surface_sh_ocean_std_ppm = Float64(dry_sh_surface.std),
        dry_surface_zonal50_std_ppm = Float64(dry_zonal_surface.std),
        dry_column_sh_ocean_mean_ppm = Float64(dry_sh_column.mean),
        dry_column_sh_ocean_std_ppm = Float64(dry_sh_column.std),
        dry_column_zonal50_std_ppm = Float64(dry_zonal_column.std),
        surface_delta_total_sh_ocean_std_ppm = surface_delta_metrics === nothing ? NaN : surface_delta_metrics.total_sh_std,
        surface_delta_tracer_sh_ocean_std_ppm = surface_delta_metrics === nothing ? NaN : surface_delta_metrics.num_sh_std,
        surface_delta_mass_sh_ocean_std_ppm = surface_delta_metrics === nothing ? NaN : surface_delta_metrics.mass_sh_std,
        surface_rel_mass_sh_ocean_std = surface_delta_metrics === nothing ? NaN : surface_delta_metrics.rel_mass_sh_std,
        surface_delta_total_zonal50_std_ppm = surface_delta_metrics === nothing ? NaN : surface_delta_metrics.total_row_std,
        surface_delta_tracer_zonal50_std_ppm = surface_delta_metrics === nothing ? NaN : surface_delta_metrics.num_row_std,
        surface_delta_mass_zonal50_std_ppm = surface_delta_metrics === nothing ? NaN : surface_delta_metrics.mass_row_std,
        surface_delta_vs_rel_mass_corr = surface_delta_metrics === nothing ? NaN : surface_delta_metrics.total_vs_rel_mass_corr,
        surface_delta_vs_mass_corr = surface_delta_metrics === nothing ? NaN : surface_delta_metrics.total_vs_mass_corr,
        surface_delta_vs_tracer_corr = surface_delta_metrics === nothing ? NaN : surface_delta_metrics.total_vs_num_corr,
        column_delta_total_sh_ocean_std_ppm = column_delta_metrics === nothing ? NaN : column_delta_metrics.total_sh_std,
        column_delta_tracer_sh_ocean_std_ppm = column_delta_metrics === nothing ? NaN : column_delta_metrics.num_sh_std,
        column_delta_mass_sh_ocean_std_ppm = column_delta_metrics === nothing ? NaN : column_delta_metrics.mass_sh_std,
        column_rel_mass_sh_ocean_std = column_delta_metrics === nothing ? NaN : column_delta_metrics.rel_mass_sh_std,
        column_delta_total_zonal50_std_ppm = column_delta_metrics === nothing ? NaN : column_delta_metrics.total_row_std,
        column_delta_tracer_zonal50_std_ppm = column_delta_metrics === nothing ? NaN : column_delta_metrics.num_row_std,
        column_delta_mass_zonal50_std_ppm = column_delta_metrics === nothing ? NaN : column_delta_metrics.mass_row_std,
        column_delta_vs_rel_mass_corr = column_delta_metrics === nothing ? NaN : column_delta_metrics.total_vs_rel_mass_corr,
        column_delta_vs_mass_corr = column_delta_metrics === nothing ? NaN : column_delta_metrics.total_vs_mass_corr,
        column_delta_vs_tracer_corr = column_delta_metrics === nothing ? NaN : column_delta_metrics.total_vs_num_corr,
        dry_surface_delta_total_sh_ocean_std_ppm = dry_surface_delta_metrics === nothing ? NaN : dry_surface_delta_metrics.total_sh_std,
        dry_surface_delta_tracer_sh_ocean_std_ppm = dry_surface_delta_metrics === nothing ? NaN : dry_surface_delta_metrics.num_sh_std,
        dry_surface_delta_mass_sh_ocean_std_ppm = dry_surface_delta_metrics === nothing ? NaN : dry_surface_delta_metrics.mass_sh_std,
        dry_surface_delta_total_zonal50_std_ppm = dry_surface_delta_metrics === nothing ? NaN : dry_surface_delta_metrics.total_row_std,
        dry_column_delta_total_sh_ocean_std_ppm = dry_column_delta_metrics === nothing ? NaN : dry_column_delta_metrics.total_sh_std,
        dry_column_delta_tracer_sh_ocean_std_ppm = dry_column_delta_metrics === nothing ? NaN : dry_column_delta_metrics.num_sh_std,
        dry_column_delta_mass_sh_ocean_std_ppm = dry_column_delta_metrics === nothing ? NaN : dry_column_delta_metrics.mass_sh_std,
        dry_column_delta_total_zonal50_std_ppm = dry_column_delta_metrics === nothing ? NaN : dry_column_delta_metrics.total_row_std,
        surface_hotspot_lon = Float64(hotspot.lon),
        surface_hotspot_lat = Float64(hotspot.lat),
        surface_hotspot_ppm = Float64(hotspot.value),
        surface_hotspot_dev_ppm = Float64(hotspot.deviation),
        threshold_surface_breach = isfinite(sh_surface.std) &&
                                   sh_surface.std > audit.threshold_surface_ppm,
        threshold_column_breach = isfinite(sh_column.std) &&
                                  sh_column.std > audit.threshold_column_ppm,
        m0_min = m0_stats === nothing ? NaN : Float64(m0_stats.min),
        m0_max = m0_stats === nothing ? NaN : Float64(m0_stats.max),
        dm_min = dm_stats === nothing ? NaN : Float64(dm_stats.min),
        dm_max = dm_stats === nothing ? NaN : Float64(dm_stats.max),
        am_absmax = Float64(am_absmax),
        bm_absmax = Float64(bm_absmax),
        cm_absmax = Float64(cm_absmax),
        cm_preclamp_absmax = Float64(cm_pre_absmax),
        cm_preclamp_gamma_max = Float64(cm_pre_gamma_max),
        cm_postclamp_gamma_max = Float64(cm_post_gamma_max),
        tm5_cm_absmax_diff = tm5_ref === nothing ? NaN : Float64(tm5_ref.cm_absmax_diff),
        tm5_cm_rms_diff = tm5_ref === nothing ? NaN : Float64(tm5_ref.cm_rms_diff),
        tm5_gamma_max = tm5_ref === nothing ? NaN : Float64(tm5_ref.gamma_max),
        tm5_gamma_ref_max = tm5_ref === nothing ? NaN : Float64(tm5_ref.gamma_ref_max),
        tm5_x_nloop_max = tm5_ref === nothing ? -1 : Int(tm5_ref.x_nloop_max),
        tm5_pit_absmax = tm5_ref === nothing ? NaN : Float64(tm5_ref.pit_absmax),
        tm5_sd_absmax = tm5_ref === nothing ? NaN : Float64(tm5_ref.sd_absmax),
    )
    push!(audit.stages, record)
    stage_idx = length(audit.stages)

    if _ll_audit_should_snapshot(audit, stage, micro, refinement, m_stats, rm_stats, c_stats)
        cm_snapshot = cm === nothing ? nothing : Array(@view Array(cm)[:, :, max(1, size(cm, 3) - 1)])
        audit.snapshots[stage_idx] = (
            surface_ppm = copy(surface_ppm),
            column_ppm = copy(column_ppm),
            surface_delta_mass_ppm = surface_delta === nothing ? nothing : copy(surface_delta.delta_mass),
            surface_delta_tracer_ppm = surface_delta === nothing ? nothing : copy(surface_delta.delta_num),
            column_delta_mass_ppm = column_delta === nothing ? nothing : copy(column_delta.delta_mass),
            m_surface = copy(Array(@view m_arr[:, :, size(m_arr, 3)])),
            cm_lowest = cm_snapshot,
        )
    end

    return nothing
end

function _ll_first_window_debug_cb(tracers, gpu, adv_ws)
    ws = _ll_massflux_workspace(adv_ws)
    ws isa MassFluxWorkspace || return nothing
    target_name = first(keys(tracers))

    return function (stage, name, rm_state, m_state)
        (name == :all || name == target_name) || return nothing
        rm_arr = rm_state isa NamedTuple ? rm_state[target_name] : rm_state
        rm_stats = _finite_array_stats(rm_arr)
        m_stats = _finite_array_stats(m_state)
        cfl_x = max_cfl_massflux_x(gpu.am, m_state, ws.cfl_x, ws.cluster_sizes)
        cfl_y = max_cfl_massflux_y(gpu.bm, m_state, ws.cfl_y)
        cfl_z = max_cfl_massflux_z(gpu.cm, m_state, ws.cfl_z)
        @info "[LL first-window debug]" stage=stage tracer=target_name rm_stats=rm_stats m_stats=m_stats cfl=(x=cfl_x, y=cfl_y, z=cfl_z)
        return nothing
    end
end

function _compose_debug_callbacks(callbacks...)
    active = filter(!isnothing, callbacks)
    isempty(active) && return nothing
    return function (stage, name, rm_state, m_state)
        for cb in active
            cb(stage, name, rm_state, m_state)
        end
        return nothing
    end
end

function _check_cm_cfl_limit!(cm_gpu, m_gpu, ws::MassFluxWorkspace{FT}, cfl_limit::FT;
                              context::AbstractString="cm", throw_on_violation::Bool=false) where FT
    cfl_z = max_cfl_massflux_z(cm_gpu, m_gpu, ws.cfl_z)
    tol = max(sqrt(eps(FT)), FT(1e-6))
    cm_top = Array(@view cm_gpu[:, :, 1])
    cm_bot = Array(@view cm_gpu[:, :, size(cm_gpu, 3)])
    top_leak = isempty(cm_top) ? zero(FT) : maximum(abs, cm_top)
    bot_leak = isempty(cm_bot) ? zero(FT) : maximum(abs, cm_bot)
    if top_leak > tol || bot_leak > tol
        msg = "Post-clamp cm boundary leak in $context: top=$top_leak bottom=$bot_leak"
        throw_on_violation ? error(msg) : @warn msg maxlog=5
    end
    if !isfinite(cfl_z) || cfl_z > cfl_limit + tol
        msg = "Post-clamp z CFL exceeded limit in $context: cfl_z=$cfl_z limit=$cfl_limit"
        throw_on_violation ? error(msg) : @warn msg maxlog=5
    end
    return cfl_z
end

# =====================================================================
# Setup helpers
# =====================================================================

"""Build diffusion workspace, dispatched on grid type for correct template array."""
function setup_diffusion_phase(model, grid::LatitudeLongitudeGrid, dt_window, sched)
    _setup_bld_workspace(model.diffusion, grid, dt_window, current_gpu(sched).m_ref)
end

function setup_diffusion_phase(model, grid::CubedSphereGrid{FT}, dt_window, sched) where FT
    AT = array_type(model.architecture)
    Hp = grid.Hp
    ref_panel = AT(zeros(FT, grid.Nc + 2Hp, grid.Nc + 2Hp, grid.Nz))
    _setup_bld_workspace(model.diffusion, grid, dt_window, ref_panel)
end

"""Return kwargs NamedTuple for IOScheduler begin_load! (physics buffers for CS)."""
function physics_load_kwargs(phys, grid::LatitudeLongitudeGrid)
    return (;)  # LL loads physics separately, not via IOScheduler
end

function physics_load_kwargs(phys, grid::CubedSphereGrid)
    # Surface field buffers: use PBL buffers if available, else dummy for tropopause
    sfc = if phys.has_pbl
        phys.pbl_sfc_cpu
    else
        (pblh=phys.troph_cpu, ustar=phys.troph_cpu,
         hflux=phys.troph_cpu, t2m=phys.troph_cpu)
    end
    return (;
        cmfmc_cpu=phys.cmfmc_cpu, dtrain_cpu=phys.dtrain_cpu,
        sfc_cpu=sfc, troph_cpu=phys.troph_cpu,
        needs_cmfmc=phys.has_conv, needs_dtrain=phys.has_dtrain,
        needs_sfc=true, needs_qv=true,
        qv_cpu=phys.qv_cpu, ps_panels=phys.ps_cpu,
        qv_next_cpu=phys.qv_next_cpu, ps_next_panels=nothing)
end

"""Log simulation start message (unified format for all grid/buffering combos)."""
function log_simulation_start(model, grid, buffering, n_win, n_sub, dw)
    grid_str = if hasproperty(grid, :Nc)
        "C$(grid.Nc)"
    else
        "$(grid.Nx)×$(grid.Ny)×$(grid.Nz)"
    end
    buf_str = nameof(typeof(buffering))

    parts = ["Starting simulation: $n_win windows × $n_sub sub-steps ($buf_str, $grid_str)"]
    if dw !== nothing && hasproperty(model.diffusion, :Kz_max)
        push!(parts, " [diffusion: Kz_max=$(model.diffusion.Kz_max), H_scale=$(model.diffusion.H_scale)]")
    end
    if _needs_pbl(model.diffusion)
        push!(parts, " [diffusion: $(_diff_label(model.diffusion)) (β_h=$(model.diffusion.β_h))]")
    end
    if _needs_convection(model.convection)
        push!(parts, " [convection: $(nameof(typeof(model.convection)))]")
    end
    @info join(parts)
end

# =====================================================================
# Dry mass helpers for LL rm↔c_dry conversions
#
# c_dry = rm / m_dry, where m_dry = m_moist × (1 - QV).
# Computed once per window; used by IC, emissions, diffusion, convection, output.
# =====================================================================

    # Mass-CFL pilot is now in src/Advection/mass_cfl_pilot.jl
    # Called from advection_phase! via Advection.find_mass_cfl_refinement()

"""Compute `phys.m_dry = m_ref × (1 - QV)` for LL grids. No-op for CS.
Skips QV correction if vertical levels don't match (e.g., merged grid)."""
compute_ll_dry_mass!(phys, sched, grid::CubedSphereGrid) = nothing
function compute_ll_dry_mass!(phys, sched, grid::LatitudeLongitudeGrid)
    gpu = current_gpu(sched)
    if phys.qv_loaded[] && size(phys.qv_gpu) == size(gpu.m_ref)
        phys.m_dry .= gpu.m_ref .* (1 .- phys.qv_gpu)
    else
        copyto!(phys.m_dry, gpu.m_ref)
    end
end

"""Return dry mass for LL rm↔c_dry conversions (pre-computed by `compute_ll_dry_mass!`)."""
ll_dry_mass(phys) = phys.m_dry

"""Recompute m_dry from evolved m_dev (post-advection) for LL output.
TM5 always uses m_evolved for mixing ratios — never m_prescribed.
The Strang-split advection evolves m_dev away from m_ref; using m_ref
for output creates noise in c = rm/m because rm was evolved with m_dev."""
compute_ll_dry_mass_evolved!(phys, sched, grid::CubedSphereGrid) = nothing
function compute_ll_dry_mass_evolved!(phys, sched, grid::LatitudeLongitudeGrid)
    gpu = current_gpu(sched)
    if phys.qv_loaded[] && size(phys.qv_gpu) == size(gpu.m_dev)
        phys.m_dry .= gpu.m_dev .* (1 .- phys.qv_gpu)
    else
        copyto!(phys.m_dry, gpu.m_dev)
    end
end

# =====================================================================
# Phase 1: Load + upload physics fields (CMFMC, DTRAIN, QV, surface)
# =====================================================================

"""
    load_and_upload_physics!(phys, sched, driver, grid, w; arch)

Load physics fields and upload to GPU. For LL, loads each field from driver
(IOScheduler only loads met data). For CS, physics was already loaded by
IOScheduler's `begin_load!` — just upload to GPU based on `sched.io_result`.
"""
function load_and_upload_physics!(phys, sched, driver,
                                   grid::LatitudeLongitudeGrid, w; arch=nothing)
    # TM5 convection fields (entu, detu, entd, detd)
    if phys.has_tm5conv
        status = load_tm5conv_window!(phys.tm5conv_cpu, driver, grid, w)
        phys.tm5conv_loaded[] = status !== false
        if phys.tm5conv_loaded[]
            for name in (:entu, :detu, :entd, :detd)
                copyto!(phys.tm5conv_gpu[name], phys.tm5conv_cpu[name])
            end
        end
    end

    # CMFMC (for Tiedtke/RAS, not used with TM5 matrix convection)
    if phys.has_conv && !phys.has_tm5conv
        status = load_cmfmc_window!(phys.cmfmc_cpu, driver, grid, w)
        phys.cmfmc_loaded[] = status !== false
        if phys.cmfmc_loaded[] && status !== :cached
            copyto!(phys.cmfmc_gpu, phys.cmfmc_cpu)
        end
    end

    # DTRAIN
    if phys.has_dtrain && phys.cmfmc_loaded[]
        status = load_dtrain_window!(phys.dtrain_cpu, driver, grid, w)
        phys.dtrain_loaded[] = status !== false
        if phys.dtrain_loaded[] && status !== :cached
            copyto!(phys.dtrain_gpu, phys.dtrain_cpu)
        end
    end

    # Invalidate RAS CFL cache on fresh data
    if phys.cmfmc_loaded[] || phys.dtrain_loaded[]
        invalidate_ras_cfl_cache!()
    end

    # Surface fields
    if phys.has_pbl
        phys.sfc_loaded[] = load_surface_fields_window!(
            phys.pbl_sfc_cpu, driver, grid, w)
        if phys.sfc_loaded[]
            copyto!(phys.pbl_sfc_gpu.pblh,  phys.pbl_sfc_cpu.pblh)
            copyto!(phys.pbl_sfc_gpu.ustar, phys.pbl_sfc_cpu.ustar)
            copyto!(phys.pbl_sfc_gpu.hflux, phys.pbl_sfc_cpu.hflux)
            copyto!(phys.pbl_sfc_gpu.t2m,   phys.pbl_sfc_cpu.t2m)
        end
    end

    # QV (specific humidity for dry-air output conversion)
    qv_status = load_qv_window!(phys.qv_cpu, driver, grid, w)
    phys.qv_loaded[] = qv_status !== false
    if phys.qv_loaded[] && qv_status !== :cached
        copyto!(phys.qv_gpu, phys.qv_cpu)
    end
end

function load_and_upload_physics!(phys, sched, driver,
                                   grid::CubedSphereGrid, w; arch=nothing)
    # Wait for async physics load (split from met load for IO overlap)
    wait_phys!(sched)
    io = sched.io_result
    io === nothing && return

    # CMFMC
    if phys.has_conv
        cmfmc_status = io.cmfmc
        phys.cmfmc_loaded[] = cmfmc_status !== false
        if phys.cmfmc_loaded[] && cmfmc_status !== :cached
            for_panels_nosync() do p
                copyto!(phys.cmfmc_gpu[p], phys.cmfmc_cpu[p])
            end
        end
    end

    # DTRAIN
    if phys.has_dtrain
        dtrain_status = io.dtrain
        phys.dtrain_loaded[] = dtrain_status !== false
        if phys.dtrain_loaded[] && dtrain_status !== :cached
            for_panels_nosync() do p
                copyto!(phys.dtrain_gpu[p], phys.dtrain_cpu[p])
            end
        end
    end

    # Invalidate RAS CFL cache on fresh data
    if (phys.cmfmc_loaded[] && io.cmfmc !== :cached) ||
       (phys.dtrain_loaded[] && io.dtrain !== :cached)
        invalidate_ras_cfl_cache!()
    end

    # QV (current window = SPHU1 in GCHP terminology)
    qv_status = io.qv
    phys.qv_loaded[] = qv_status !== false
    if phys.qv_loaded[] && qv_status !== :cached
        for_panels_nosync() do p
            copyto!(phys.qv_gpu[p], phys.qv_cpu[p])
        end
        fill_panel_halos!(phys.qv_gpu, grid)
    end

    # QV next window (= SPHU2 in GCHP terminology)
    # Used for target PE in dry-basis remap: dp_dry_target = DELP_next × (1-QV_next)
    if hasproperty(io, :qv_next_from_v4) && io.qv_next_from_v4
        # v4 binary: QV_end already loaded atomically into qv_next_cpu by load_all_window!
        phys.qv_next_loaded[] = true
        for_panels_nosync() do p
            copyto!(phys.qv_next_gpu[p], phys.qv_next_cpu[p])
        end
        fill_panel_halos!(phys.qv_next_gpu, grid)
    else
        # Fallback: load from separate CTM_I1/I3 file
        n_win = total_windows(driver)
        if w < n_win
            qv_next_status = load_qv_window!(phys.qv_next_cpu, driver, grid, w + 1)
            phys.qv_next_loaded[] = qv_next_status !== false
            if phys.qv_next_loaded[] && qv_next_status !== :cached
                for_panels_nosync() do p
                    copyto!(phys.qv_next_gpu[p], phys.qv_next_cpu[p])
                end
                fill_panel_halos!(phys.qv_next_gpu, grid)
            end
        else
            phys.qv_next_loaded[] = false
        end
    end

    # Surface fields
    sfc_status = io.sfc
    phys.sfc_loaded[] = sfc_status !== false
    phys.troph_loaded[] = phys.sfc_loaded[]
    if phys.sfc_loaded[] && phys.has_pbl
        for_panels_nosync() do p
            copyto!(phys.pbl_sfc_gpu.pblh[p],  phys.pbl_sfc_cpu.pblh[p])
            copyto!(phys.pbl_sfc_gpu.ustar[p], phys.pbl_sfc_cpu.ustar[p])
            copyto!(phys.pbl_sfc_gpu.hflux[p], phys.pbl_sfc_cpu.hflux[p])
            copyto!(phys.pbl_sfc_gpu.t2m[p],   phys.pbl_sfc_cpu.t2m[p])
        end
    end

end

"""Compute PS from DELP (CS only). Called after begin_load! — reads curr_cpu (safe)."""
compute_ps_phase!(phys, sched, grid::LatitudeLongitudeGrid) = nothing
compute_ps_phase!(phys, sched, grid::CubedSphereGrid) = _compute_ps_from_delp!(phys, sched, grid)

"""Compute surface pressure from DELP for CS grids (no-op if PS loaded from binary)."""
function _compute_ps_from_delp!(phys, sched, grid::CubedSphereGrid{FT}) where FT
    Nc, Nz, Hp = grid.Nc, grid.Nz, grid.Hp
    ps_cpu = phys.ps_cpu
    _ps_from_bin = phys.sfc_loaded[] && !iszero(ps_cpu[1][Hp + 1, Hp + 1])
    _ps_from_bin && return

    cpu = current_cpu(sched)
    for p in 1:6
        fill!(ps_cpu[p], zero(FT))
        delp_p = cpu.delp[p]
        @inbounds for k in 1:Nz
            for jj in 1:Nc, ii in 1:Nc
                ps_cpu[p][Hp + ii, Hp + jj] += delp_p[Hp + ii, Hp + jj, k]
            end
        end
    end
end

# =====================================================================
# Phase 2: Process met after upload (scale fluxes, compute DELP for LL)
# =====================================================================

"""Process met data after GPU upload. LL: handle raw met + compute DELP. CS: scale fluxes."""
function process_met_after_upload!(sched, phys, grid::LatitudeLongitudeGrid{FT},
                                    driver, half_dt; use_gchp::Bool=false) where FT
    gpu = current_gpu(sched)
    cpu = current_cpu(sched)

    if driver isa AbstractRawMetDriver
        copyto!(gpu.Δp, gpu.m_ref)
        compute_air_mass!(gpu.m_ref, gpu.Δp, grid)
        copyto!(gpu.m_dev, gpu.m_ref)
        copyto!(gpu.u, gpu.am)
        copyto!(gpu.v, gpu.bm)
        compute_mass_fluxes!(gpu.am, gpu.bm, gpu.cm,
                              gpu.u, gpu.v, grid, gpu.Δp, half_dt)
    elseif phys.needs_delp
        _compute_delp_ll!(gpu, cpu, phys, grid)
    end
    # Preprocessed binary: am/bm/cm are per spectral half_dt and used directly.
    # NO runtime scaling — the preprocessor stores fluxes per half-step (450s),
    # and n_sub Strang cycles accumulate the correct total:
    #   n_sub × 2 × am = steps_per_met × 2 × half_dt = window_dt
    # This matches the CS convention (no /n_sub, no scaling) and the old LL code.

    # TM5 convention: j=1,Ny are real cells at ±89.75°.
    # GPU reduced-grid clustering now supports full TM5 sizes (_MAX_GPU_CLUSTER=720).
    # Zero am at pole rows as temporary guard until mass-CFL pilot is validated.
    # bm at pole FACES (j=1, j=Ny+1) is already zero from the preprocessor.
    Ny = grid.Ny
    @views gpu.am[:, 1, :]  .= zero(FT)
    @views gpu.am[:, Ny, :] .= zero(FT)
end

function process_met_after_upload!(sched, phys, grid::CubedSphereGrid{FT},
                                    driver, half_dt; use_gchp::Bool=false) where FT
    gpu = current_gpu(sched)
    if use_gchp
        # GCHP path: leave AM/BM/CX/CY/XFX/YFX UNSCALED.
        # AM/BM are in kg/s; conversion to Pa·m² happens in gchp_offline_advection_phase!.
        # CX/CY/XFX/YFX stay at full accumulated values; subcycling divides them.
        return
    end
    for_panels_nosync() do p
        gpu.am[p] .*= half_dt
        gpu.bm[p] .*= half_dt
    end
    if gpu.cx !== nothing
        for_panels_nosync() do p
            gpu.cx[p]  .*= FT(0.5)
            gpu.cy[p]  .*= FT(0.5)
        end
    end
    if gpu.xfx !== nothing
        for_panels_nosync() do p
            gpu.xfx[p] .*= FT(0.5)
            gpu.yfx[p] .*= FT(0.5)
        end
    end
end

"""Compute DELP for LL from air mass: Δp = m × g / area."""
function _compute_delp_ll!(gpu, cpu, phys, grid::LatitudeLongitudeGrid{FT}) where FT
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    g = FT(grid.gravity)
    delp_cpu = phys.delp_cpu
    area_j = phys.area_j
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        delp_cpu[i, j, k] = cpu.m[i, j, k] * g / area_j[j]
    end
    copyto!(gpu.Δp, delp_cpu)
end

# =====================================================================
# Phase 3: Compute air mass from DELP
# =====================================================================

"""Compute air mass. LL: already in m_ref after upload. CS: from DELP per panel (dry when QV available and dry_correction enabled)."""
compute_air_mass_phase!(sched, air, phys, grid::LatitudeLongitudeGrid, gc; dry_correction::Bool=true) = nothing

function compute_air_mass_phase!(sched, air, phys, grid::CubedSphereGrid, gc; dry_correction::Bool=true)
    gpu = current_gpu(sched)
    Nc, Nz, Hp = grid.Nc, grid.Nz, grid.Hp
    if dry_correction && phys.qv_loaded[]
        for_panels_nosync() do p
            compute_air_mass_panel!(air.m[p], gpu.delp[p], phys.qv_gpu[p],
                                    gc.area[p], gc.gravity, Nc, Nz, Hp)
        end
    else
        for_panels_nosync() do p
            compute_air_mass_panel!(air.m[p], gpu.delp[p],
                                    gc.area[p], gc.gravity, Nc, Nz, Hp)
        end
    end
    # Always compute MOIST air mass for convection (m_wet = delp × area / g).
    # Convective mass fluxes (CMFMC, DTRAIN) are on MOIST basis, so rm↔q
    # conversion must use moist air mass for consistency (GCHP convention).
    for_panels_nosync() do p
        compute_air_mass_panel!(air.m_wet[p], gpu.delp[p],
                                gc.area[p], gc.gravity, Nc, Nz, Hp)
    end
end

# =====================================================================
# Phase 4: IC finalization + reference mass save
# =====================================================================

"""Finalize deferred initial conditions (first window only).
LL: after vertical interp sets VMR c, convert to tracer mass rm = c × m_ref.
TM5-faithful: transport on moist basis, dry correction at output only."""
function finalize_ic_phase!(tracers, sched, air, phys, grid::LatitudeLongitudeGrid)
    gpu = current_gpu(sched)
    if has_deferred_ic_vinterp()
        qv_ic = phys.qv_loaded[] && size(phys.qv_gpu) == size(gpu.m_ref) ? phys.qv_gpu : nothing
        finalize_ic_vertical_interp!(tracers, gpu.m_ref, grid; qv_3d=qv_ic)
    end
    # Convert VMR → tracer mass: rm = c × m_ref (moist basis, like TM5)
    for (_, c) in pairs(tracers)
        c .*= gpu.m_ref
    end
end

function finalize_ic_phase!(tracers, sched, air, phys, grid::CubedSphereGrid)
    has_deferred_ic_vinterp() || return
    gpu = current_gpu(sched)
    # IC values are dry VMR. air.m is already dry when QV available, moist
    # otherwise. Either way: rm = q × m gives correct tracer mass.
    finalize_ic_vertical_interp!(tracers, air.m, gpu.delp, grid; qv_panels=nothing)
end

"""Save reference air mass. LL: no-op (m_ref is the reference). CS: copy m → m_ref."""
save_reference_mass!(sched, ::Nothing, ::LatitudeLongitudeGrid) = nothing

function save_reference_mass!(sched, air, grid::CubedSphereGrid)
    for_panels_nosync() do p
        copyto!(air.m_ref[p], air.m[p])
    end
end

# =====================================================================
# Phase 5: Compute vertical mass flux (cm)
# =====================================================================

"""Compute cm. LL: already in met buffer. CS: from continuity of am/bm."""
compute_cm_phase!(sched, air, phys, grid::LatitudeLongitudeGrid, gc;
                   has_next::Bool=false, n_sub::Int=1) = nothing

function compute_cm_phase!(sched, air, phys, grid::CubedSphereGrid, gc;
                            has_next::Bool=false, n_sub::Int=1)
    gpu = current_gpu(sched)
    Nc, Nz = grid.Nc, grid.Nz

    # Pure mass-flux closure: cm from continuity of dry am/bm.
    # Dry air mass is conserved (no sources/sinks), so the dry horizontal
    # flux divergence exactly determines the dry vertical flux.
    for_panels_nosync() do p
        compute_cm_panel!(gpu.cm[p], gpu.am[p], gpu.bm[p], gc.bt, Nc, Nz)
    end
end

"""Wait for next-window DELP and upload (CS DoubleBuffer pressure fixer)."""
wait_and_upload_next_delp!(::IOScheduler{SingleBuffer}, ::Any) = nothing
wait_and_upload_next_delp!(::IOScheduler, ::LatitudeLongitudeGrid) = nothing

function wait_and_upload_next_delp!(sched::IOScheduler{DoubleBuffer},
                                     grid::CubedSphereGrid)
    # Wait for met-only task (fast: DELP + fluxes). Physics task runs independently.
    if sched.load_task !== nothing
        fetch(sched.load_task)
        sched.load_task = nothing
    end
    nc = next_cpu(sched)
    ng = next_gpu(sched)
    for_panels_nosync() do p
        copyto!(ng.delp[p], nc.delp[p])
    end
end

# =====================================================================
# Phase 6: CFL diagnostic
# =====================================================================

"""Update CFL diagnostic string (CS only, periodic)."""
update_cfl_diagnostic!(diag, sched, air, grid::LatitudeLongitudeGrid, ws, w) = nothing

function update_cfl_diagnostic!(diag, sched, air, grid::CubedSphereGrid, ws, w)
    (w == 1 || w % 24 == 0) || return
    gpu = current_gpu(sched)
    Hp = grid.Hp
    cfl_x = maximum(max_cfl_x_cs(gpu.am[p], air.m_ref[p], ws.cfl_x, Hp) for p in 1:6)
    cfl_y = maximum(max_cfl_y_cs(gpu.bm[p], air.m_ref[p], ws.cfl_y, Hp) for p in 1:6)
    diag.cfl_value = @sprintf("x=%.3f y=%.3f", cfl_x, cfl_y)
end

# =====================================================================
# Phase 7: Apply emissions
# =====================================================================

"""Apply emissions, dispatched on grid type.
LL tracers are rm — convert to VMR for emission kernels, then back (same m_ref → exact).
TM5-faithful: moist basis for boundary conversions, dry correction at output only."""
function apply_emissions_phase!(tracers, emi_state, sched, phys, gc,
                                 grid::LatitudeLongitudeGrid, dt_window;
                                 sim_hours::Float64=0.0, arch=nothing)
    emi_data, area_j_dev, A_coeff, B_coeff = emi_state
    gpu = current_gpu(sched)
    # rm → c for emission kernels (moist basis, like TM5)
    for (_, rm) in pairs(tracers)
        rm ./= gpu.m_ref
    end
    _apply_emissions_latlon!(tracers, emi_data, area_j_dev,
                              gpu.ps, A_coeff, B_coeff, grid, dt_window;
                              sim_hours, arch,
                              delp=(phys.sfc_loaded[] ? gpu.Δp : nothing),
                              pblh=(phys.sfc_loaded[] && phys.pbl_sfc_gpu !== nothing ?
                                    phys.pbl_sfc_gpu.pblh : nothing))
    # c → rm
    for (_, c) in pairs(tracers)
        c .*= gpu.m_ref
    end
end

function apply_emissions_phase!(tracers, emi_state, sched, phys, gc,
                                 grid::CubedSphereGrid, dt_window;
                                 sim_hours::Float64=0.0, arch=nothing)
    emi_data = emi_state[1]
    gpu = current_gpu(sched)
    Nc, Hp = grid.Nc, grid.Hp
    _apply_emissions_cs!(tracers, emi_data, gc.area, dt_window, Nc, Hp;
                          sim_hours, arch,
                          delp=gpu.delp,
                          pblh=(phys.pbl_sfc_gpu !== nothing ?
                                phys.pbl_sfc_gpu.pblh : nothing))
end

# =====================================================================
# Phase 8: Advection sub-stepping (+ convection per sub-step)
# =====================================================================

"""
    advection_phase!(tracers, sched, air, phys, model, grid, ws, n_sub, dt_sub, step)

Recompute cm from horizontal divergence of am/bm on CPU.
Simple top-down accumulation per column. Called once per substep when interpolating (v4).
cm[1] = 0 (TOA), cm[Nz+1] = residual → enforced to zero at boundaries.
"""
function _compute_cm_from_divergence_gpu!(cm_gpu, am_gpu, bm_gpu, m_gpu, grid)
    cm = Array(cm_gpu)
    am = Array(am_gpu)
    bm = Array(bm_gpu)
    FT = eltype(cm)
    Nx, Ny = grid.Nx, grid.Ny
    Nz = size(cm, 3) - 1
    fill!(cm, zero(FT))

    # Use B-coefficient correction (TM5 dynam0 formula) if available
    B_ifc = hasproperty(grid.vertical, :B) ? Float64.(grid.vertical.B) : Float64[]

    if !isempty(B_ifc) && length(B_ifc) == Nz + 1
        @inbounds for j in 1:Ny, i in 1:Nx
            pit = 0.0
            for k in 1:Nz
                pit += (Float64(am[i+1, j, k]) - Float64(am[i, j, k])) +
                       (Float64(bm[i, j+1, k]) - Float64(bm[i, j, k]))
            end
            acc = 0.0
            for k in 1:Nz
                div_h = (Float64(am[i+1, j, k]) - Float64(am[i, j, k])) +
                        (Float64(bm[i, j+1, k]) - Float64(bm[i, j, k]))
                acc = acc - div_h + (B_ifc[k+1] - B_ifc[k]) * pit
                cm[i, j, k+1] = FT(acc)
            end
        end
    else
        @inbounds for j in 1:Ny, i in 1:Nx
            acc = 0.0
            for k in 1:Nz
                div_h = (Float64(am[i+1, j, k]) - Float64(am[i, j, k])) +
                        (Float64(bm[i, j+1, k]) - Float64(bm[i, j, k]))
                acc -= div_h
                cm[i, j, k+1] = FT(acc)
            end
        end
    end
    @views cm[:, :, 1]   .= zero(FT)
    @views cm[:, :, end] .= zero(FT)
    copyto!(cm_gpu, cm)
end

"""
Clamp interior cm faces so the donor-based z-CFL stays below `cfl_limit`,
and enforce impermeable top/bottom boundaries. Because each interior face is
shared by the two adjacent levels and cm[:,:,1/end] are reset to zero, the
resulting z update remains exactly column-conservative.

ERA5 spectral data has ~0.02% of cells with extreme Z-CFL from deep convection.
Operates on CPU buffers before GPU upload (called once per window).
"""
function _clamp_cm_cfl!(cm_gpu, m_gpu, cfl_limit)
    cm = Array(cm_gpu)
    m = Array(m_gpu)
    Nx, Ny, Nz = size(m)
    FT = eltype(cm)
    n_clamped = 0
    n_boundary_zeroed = 0
    @inbounds for j in 1:Ny, i in 1:Nx
        if cm[i, j, 1] != zero(FT)
            cm[i, j, 1] = zero(FT)
            n_boundary_zeroed += 1
        end
        if cm[i, j, Nz + 1] != zero(FT)
            cm[i, j, Nz + 1] = zero(FT)
            n_boundary_zeroed += 1
        end
    end
    @inbounds for k in 2:Nz, j in 1:Ny, i in 1:Nx
        # cm[i,j,k] is flux at interface between level k-1 (above) and k (below)
        m_above = m[i, j, k-1]
        m_below = m[i, j, k]
        cm_val = cm[i, j, k]
        if cm_val > zero(FT)
            # Positive cm moves mass from k-1 into k; donor is the upper level.
            max_flux = cfl_limit * m_above
        else
            # Negative cm moves mass from k into k-1; donor is the lower level.
            max_flux = cfl_limit * m_below
        end
        if abs(cm_val) > max_flux
            cm[i, j, k] = sign(cm_val) * max_flux
            n_clamped += 1
        end
    end
    if n_clamped > 0 || n_boundary_zeroed > 0
        copyto!(cm_gpu, cm)
        @info "Clamped $n_clamped interior cm faces ($(round(n_clamped/(Nx*Ny*Nz)*100, digits=4))%) to CFL<$cfl_limit; zeroed $n_boundary_zeroed boundary faces" maxlog=5
    end
end

"""
Advection + convection sub-stepping. LL: NamedTuple dispatch (all tracers together).
CS: per-tracer advection with mass fixer + m_ref advance along pressure trajectory.
"""
function advection_phase!(tracers, sched, air, phys, model,
                           grid::LatitudeLongitudeGrid{FT},
                           ws, n_sub, dt_sub, step;
                           ws_lr=nothing, ws_vr=nothing, gc=nothing,
                           geom_gchp=nothing, ws_gchp=nothing,
                           has_next::Bool=false) where FT
    gpu = current_gpu(sched)
    # Build advection workspace — for Prather, augment with per-tracer slope storage
    adv_ws = _build_advection_workspace(gpu.ws, model.advection_scheme, tracers, gpu.m_ref)
    if step[] == 0 && hasproperty(adv_ws, :prather)
        for pw in values(adv_ws.prather)
            pw.initialized[] = false
        end
    end
    mass_ws = _ll_massflux_workspace(adv_ws)
    cfl_limit = FT(0.95)
    current_window = fld(step[], n_sub) + 1
    debug_window = _ll_debug_window(model)
    debug_this_window = (_ll_debug_first_window(model) && current_window == 1) ||
                        (debug_window !== nothing && current_window == debug_window)
    debug_all_substeps = debug_this_window && _ll_debug_all_substeps(model)
    use_mass_cfl_pilot = _ll_use_mass_cfl_pilot(model)
    max_mass_cfl_refinement = _ll_mass_cfl_pilot_max_refinement(model)
    audit = LL_AUDIT[]
    if audit isa LLTransportAudit
        audit.metadata["n_sub"] = n_sub
        audit.metadata["current_window"] = current_window
    end

    # TM5-style temporal interpolation (v4 binaries with flux deltas).
    # Each substep uses linearly interpolated fluxes and prescribed m_target.
    # Without deltas (v3): use constant fluxes with cm clamp.
    has_deltas = _ll_use_flux_delta(model, gpu)
    if !has_deltas
        _clamp_cm_cfl!(gpu.cm, gpu.m_ref, cfl_limit)
        if debug_this_window
            _check_cm_cfl_limit!(gpu.cm, gpu.m_ref, mass_ws, cfl_limit;
                                 context="window-start", throw_on_violation=true)
        end
    end

    debug_cb = debug_this_window ?
        _ll_first_window_debug_cb(tracers, gpu, adv_ws) : nothing

    # Save uninterpolated fluxes (needed to compute per-substep interpolation)
    # When has_deltas=true, we modify gpu.am/bm/cm in-place each substep.
    # They'll be overwritten by upload_met! at the next window.
    am0 = has_deltas ? copy(gpu.am) : gpu.am
    bm0 = has_deltas ? copy(gpu.bm) : gpu.bm
    cm0 = has_deltas ? copy(gpu.cm) : gpu.cm
    m0  = has_deltas ? copy(gpu.m_ref) : nothing

    copyto!(gpu.m_dev, gpu.m_ref)
    if _ll_audit_enabled(audit, current_window, step[] + 1)
        _ll_audit_stage!(audit, "window_start", grid, tracers, gpu.m_dev;
                         window=current_window, dyn_interval=step[] + 1,
                         substep=0, micro=0, refinement=1,
                         time_fraction=0.0, qv_loaded=phys.qv_loaded[],
                         has_deltas=has_deltas, cm=gpu.cm,
                         qv=phys.qv_loaded[] ? phys.qv_gpu : nothing)
    end
    for s in 1:n_sub
        step[] += 1

        t = FT(s - FT(0.5)) / FT(n_sub)
        cm_preclamp = nothing
        if has_deltas
            # Interpolate fluxes to substep midpoint (TM5 TimeInterpolation)
            gpu.am .= am0 .+ t .* gpu.dam
            gpu.bm .= bm0 .+ t .* gpu.dbm
            # Re-apply pole zeroing (dam/dbm may have non-zero polar values)
            Ny = grid.Ny
            @views gpu.am[:, 1, :]    .= zero(FT)
            @views gpu.am[:, Ny, :]   .= zero(FT)
            @views gpu.bm[:, 1, :]    .= zero(FT)
            @views gpu.bm[:, Ny+1, :] .= zero(FT)
            # Prescribe m_target along pressure trajectory (GEOS approach)
            gpu.m_dev .= m0 .+ t .* gpu.dm
            # Recompute cm from interpolated am/bm constrained by m_target
            _compute_cm_from_divergence_gpu!(gpu.cm, gpu.am, gpu.bm, gpu.m_dev, grid)
            if _ll_audit_enabled(audit, current_window, step[])
                cm_preclamp = copy(gpu.cm)
            end
            _clamp_cm_cfl!(gpu.cm, gpu.m_dev, cfl_limit)
            if debug_cb !== nothing
                _check_cm_cfl_limit!(gpu.cm, gpu.m_dev, mass_ws, cfl_limit;
                                     context="window=$(step[] + 1) substep=$s",
                                     throw_on_violation=true)
            end
        end

        refinement = 1
        if use_mass_cfl_pilot
            refinement = find_mass_cfl_refinement(gpu.m_dev, gpu.am, gpu.bm, gpu.cm,
                                                  grid, mass_ws.cluster_sizes, 1;
                                                  beta_limit=cfl_limit,
                                                  max_r=max_mass_cfl_refinement)
            if refinement > 1
                @info "[LL mass-CFL pilot]" window=step[] substep=s refinement=refinement maxlog=10
                scale = FT(refinement)
                gpu.am ./= scale
                gpu.bm ./= scale
                gpu.cm ./= scale
            end
        end

        if _ll_audit_enabled(audit, current_window, step[])
            _ll_audit_stage!(audit, "substep_setup", grid, tracers, gpu.m_dev;
                             window=current_window, dyn_interval=step[],
                             substep=s, micro=0, refinement=refinement,
                             time_fraction=t, qv_loaded=phys.qv_loaded[],
                             has_deltas=has_deltas, am=gpu.am, bm=gpu.bm, cm=gpu.cm,
                             cm_preclamp=cm_preclamp, m0=m0, dm=gpu.dm,
                             cluster_sizes=mass_ws.cluster_sizes,
                             qv=phys.qv_loaded[] ? phys.qv_gpu : nothing)
        end

        for micro in 1:refinement
            log_debug_cb = if debug_cb !== nothing &&
                              (debug_all_substeps || s == 1) &&
                              (micro == 1 || (debug_all_substeps && micro == refinement))
                let parent = debug_cb,
                    substep_label = "substep=$s",
                    micro_label = refinement > 1 ? "micro=$micro/$refinement" : "micro=1/1"
                    (stage, name, rm_state, m_state) -> parent("$(substep_label)/$(micro_label)/$(stage)", name, rm_state, m_state)
                end
            else
                nothing
            end
            audit_debug_cb = if _ll_audit_enabled(audit, current_window, step[])
                let audit_ref = audit,
                    tracer_name = audit.tracer,
                    dyn_interval = step[],
                    substep = s,
                    micro_idx = micro,
                    refinement_local = refinement,
                    qv_loaded = phys.qv_loaded[],
                    has_deltas_local = has_deltas,
                    t_local = t,
                    qv_state = phys.qv_loaded[] ? phys.qv_gpu : nothing
                    function (stage, name, rm_state, m_state)
                        (name == :all || name == tracer_name) || return nothing
                        _ll_audit_stage!(audit_ref, stage, grid, rm_state, m_state;
                                         window=current_window, dyn_interval=dyn_interval,
                                         substep=substep, micro=micro_idx,
                                         refinement=refinement_local,
                                         time_fraction=t_local, qv_loaded=qv_loaded,
                                         has_deltas=has_deltas_local,
                                         qv=qv_state)
                        return nothing
                    end
                end
            else
                nothing
            end
            wrapped_debug_cb = _compose_debug_callbacks(log_debug_cb, audit_debug_cb)
            _apply_advection_latlon!(tracers, gpu.m_dev,
                                     gpu.am, gpu.bm, gpu.cm,
                                     grid, model.advection_scheme, adv_ws;
                                     cfl_limit=cfl_limit,
                                     debug_cb=wrapped_debug_cb)
        end

        if refinement > 1
            scale = FT(refinement)
            gpu.am .*= scale
            gpu.bm .*= scale
            gpu.cm .*= scale
        end
    end

    # Set end-of-window mass. For v4, reset both reference and working mass
    # to the prescribed window endpoint before convection/output.
    if has_deltas
        gpu.m_ref .= m0 .+ gpu.dm
        copyto!(gpu.m_dev, gpu.m_ref)
    else
        copyto!(gpu.m_ref, gpu.m_dev)
    end

    # Convective transport ONCE per window (after all substeps).
    # rm↔c roundtrip uses m_ref.
    dt_conv = FT(n_sub) * dt_sub
    _has_conv = (phys.has_tm5conv && phys.tm5conv_loaded[]) || phys.cmfmc_loaded[]
    if _has_conv
        for (_, rm) in pairs(tracers)
            rm ./= gpu.m_ref
        end
        if phys.has_tm5conv && phys.tm5conv_loaded[]
            # TM5 matrix convection — GPU path via KA kernels.
            # Uses pre-allocated workspace for the per-column transfer matrix.
            convect!(tracers, phys.tm5conv_gpu, gpu.Δp,
                      model.convection, grid, dt_conv, phys.planet;
                      workspace=phys.tm5conv_ws)
        else
            convect!(tracers, phys.cmfmc_gpu, gpu.Δp,
                      model.convection, grid, dt_conv, phys.planet;
                      dtrain_panels=phys.dtrain_loaded[] ? phys.dtrain_gpu : nothing,
                      workspace=phys.ras_workspace)
        end
        for (_, c) in pairs(tracers)
            c .*= gpu.m_ref
        end
    end
end

# Lazy-allocated Prather workspace caches (avoids re-allocation per window)
const _PRATHER_WS_CACHE = Ref{Any}(nothing)
const _CS_PRATHER_WS_CACHE = Ref{Any}(nothing)

_build_advection_workspace(ws, scheme, tracers, m) = ws  # default: return MassFluxWorkspace as-is
function _build_advection_workspace(ws, scheme::PratherAdvection, tracers, m)
    if _PRATHER_WS_CACHE[] === nothing
        _PRATHER_WS_CACHE[] = allocate_prather_workspaces(tracers, m)
        @info "Allocated Prather workspaces for $(length(tracers)) tracers"
    end
    return (; base=ws, prather=_PRATHER_WS_CACHE[])
end

"""Get per-tracer CS Prather workspace (lazy-allocated)."""
function _get_cs_prather_ws(tracers, grid, arch)
    if _CS_PRATHER_WS_CACHE[] === nothing
        _CS_PRATHER_WS_CACHE[] = allocate_cs_prather_workspaces(tracers, grid, arch)
        @info "Allocated CS Prather workspaces for $(length(tracers)) tracers"
    end
    return _CS_PRATHER_WS_CACHE[]
end

# ---------------------------------------------------------------------------
# GCHP advection: dry-basis implementation
# ---------------------------------------------------------------------------
function _gchp_advection_dry!(tracers, sched, air, phys, model,
        grid::CubedSphereGrid{FT}, ws, ws_lr, ws_vr, gc,
        n_sub, dt_sub, step, _ORD, has_next) where FT
    gpu = current_gpu(sched)
    Nc, Nz, Hp = grid.Nc, grid.Nz, grid.Hp
    N = Nc + 2Hp
    cx_gpu, cy_gpu = gpu.cx, gpu.cy
    g_val = FT(grid.gravity)

    step[] += n_sub

    # Step 1: rm → q (dry VMR)
    for (_, rm_t) in pairs(tracers)
        rm_to_q_panels!(rm_t, air.m, grid)
    end

    # Pre-compute constants
    mfx_dt = FT(model.met_data.mass_flux_dt)
    am_to_mfx = g_val * mfx_dt
    inv_am_to_mfx = FT(1) / am_to_mfx
    rarea = ntuple(p -> FT(1) ./ gc.area[p], 6)

    n_loop = has_next ? n_sub : 1
    per_step = model.advection_scheme.per_step_remap

    # QV_next for target PE / scaling (used in both paths)
    qv_tgt = phys.qv_next_loaded[] ? phys.qv_next_gpu :
             phys.qv_loaded[] ? phys.qv_gpu : nothing

    if has_next && phys.qv_loaded[] && per_step
        # ── Per-substep remap: matches GCHP offline_tracer_advection ────────
        # Each 450s step: horizontal transport → source PE → target PE
        # (hybrid from evolved PS) → remap → scale → rm→q for next step.
        ng = next_gpu(sched)

        for _isub in 1:n_loop
            frac_start = FT(_isub - 1) / FT(n_loop)
            frac_end   = FT(_isub)     / FT(n_loop)

            # Reset dp_work to interpolated dry DELP at start of substep
            for_panels_nosync() do p
                be = get_backend(ws_vr.dp_work[p])
                _interpolate_dry_dp_kernel!(be, 256)(ws_vr.dp_work[p], gpu.delp[p], ng.delp[p],
                          phys.qv_gpu[p], qv_tgt !== nothing ? qv_tgt[p] : phys.qv_gpu[p],
                          frac_start; ndrange=(N, N, Nz))
            end

            # Horizontal advection: dp_work → evolved dpA
            for_panels_nosync() do p
                gpu.am[p] .*= am_to_mfx
                gpu.bm[p] .*= am_to_mfx
            end
            gchp_tracer_2d!(tracers, ws_vr.dp_work, gpu.am, gpu.bm,
                              cx_gpu, cy_gpu, gpu.xfx, gpu.yfx,
                              gc.area, rarea, grid, _ORD, ws_lr, ws_vr.m_save)
            for_panels_nosync() do p
                gpu.am[p] .*= inv_am_to_mfx
                gpu.bm[p] .*= inv_am_to_mfx
            end

            # Source PE from evolved dpA (dp_work after horizontal step)
            for_panels_nosync() do p
                compute_air_mass_panel!(ws_vr.m_save[p], ws_vr.dp_work[p],
                                        gc.area[p], g_val, Nc, Nz, Hp)
            end
            compute_source_pe_from_evolved_mass!(ws_vr, ws_vr.m_save, gc, grid)

            # Target PE: direct cumsum from prescribed dp, with surface PE locked
            # to source (evolved) surface PE. This prevents column mass change
            # through the remap while keeping interior levels at prescribed positions.
            # GCHP uses hybrid PE + surface locking, but hybrid needs moist PS.
            # For dry basis: direct cumsum + surface lock + bottom-layer absorption.
            if _isub < n_loop
                # Target PE: direct cumsum from prescribed dp, with column mass
                # scaled to match source (evolved) surface pressure.
                # This distributes the mass adjustment proportionally across ALL
                # levels (not just the bottom layer, which caused q distortion).
                for_panels_nosync() do p
                    be = get_backend(ws_vr.dp_work[p])
                    _interpolate_dry_dp_kernel!(be, 256)(ws_vr.dp_work[p], gpu.delp[p], ng.delp[p],
                              phys.qv_gpu[p], qv_tgt !== nothing ? qv_tgt[p] : phys.qv_gpu[p],
                              frac_end; ndrange=(N, N, Nz))
                end
                compute_target_pressure_from_delp_direct!(ws_vr, ws_vr.dp_work, gc, grid)
                # Scale dp_tgt so column sum = source column sum (ps_src from evolved mass).
                # dp_tgt[k] *= ps_src / ps_tgt (proportional distribution).
                for_panels_nosync() do p
                    be = get_backend(ws_vr.pe_tgt[p])
                    _scale_dp_tgt_to_source_ps_kernel!(be, 256)(
                        ws_vr.pe_tgt[p], ws_vr.dp_tgt[p],
                        ws_vr.pe_src[p], ws_vr.ps_src[p],
                        Nc, Nz; ndrange=(Nc, Nc))
                end
            else
                # Last: target = ng.delp × (1-qv_tgt) — identical to single-remap
                if qv_tgt !== nothing
                    compute_target_pressure_from_dry_delp_direct!(ws_vr, ng.delp,
                        qv_tgt, gc, grid)
                else
                    compute_target_pressure_from_delp_direct!(ws_vr, ng.delp, gc, grid)
                end
            end

            # q → rm, remap, fillz
            for (_, rm_t) in pairs(tracers)
                q_to_rm_panels!(rm_t, ws_vr.m_save, grid)
            end
            for (_, rm_t) in pairs(tracers)
                vertical_remap_cs!(rm_t, ws_vr.m_save, ws_vr, ws, gc, grid; hybrid_pe=true)
            end
            for (_, rm_t) in pairs(tracers)
                fillz_panels!(rm_t, ws_vr.dp_tgt, grid)
            end

            # Scale and prepare for next substep
            if _isub < n_loop
                # No intermediate scaling — proportional dp_tgt scaling + surface-locked
                # PE ensures column mass conservation. Float32 PPM drift is ~0.002%/2d.

                # Convert rm → q using TARGET air mass (surface-locked dp_tgt).
                # Using prescribed dp_work would cause mass loss because the
                # prescribed column mass ≠ surface-locked column mass.
                # Write dp_tgt back to dp_work interior for compute_air_mass_panel!.
                for_panels_nosync() do p
                    be = get_backend(ws_vr.dp_work[p])
                    _copy_dp_tgt_to_dp_work_kernel!(be, 256)(
                        ws_vr.dp_work[p], ws_vr.dp_tgt[p], Hp, Nc, Nz;
                        ndrange=(Nc, Nc, Nz))
                end
                for_panels_nosync() do p
                    compute_air_mass_panel!(ws_vr.m_save[p], ws_vr.dp_work[p],
                                            gc.area[p], g_val, Nc, Nz, Hp)
                end
                for (_, rm_t) in pairs(tracers)
                    rm_to_q_panels!(rm_t, ws_vr.m_save, grid)
                end
            else
                # Last substep: scale against ng.delp × (1-qv_tgt)
                for (tname, rm_t) in pairs(tracers)
                    scaling = gchp_calc_scaling_factor(rm_t, ws_vr.dp_tgt, ng.delp,
                                  gc, grid; qv_panels=qv_tgt)
                    @info "calcScaling: $tname = $scaling" maxlog=200
                    apply_scaling_factor!(rm_t, scaling, grid)
                end
                # tracers remain in rm form after last substep
            end
        end

    elseif has_next && phys.qv_loaded[]
        # ── n_sub loop, single remap at end (original behavior) ─────────────
        ng = next_gpu(sched)
        for _isub in 1:n_loop
            frac = FT(_isub - 1) / FT(n_loop)
            for_panels_nosync() do p
                be = get_backend(ws_vr.dp_work[p])
                _interpolate_dry_dp_kernel!(be, 256)(ws_vr.dp_work[p], gpu.delp[p], ng.delp[p],
                          phys.qv_gpu[p], qv_tgt !== nothing ? qv_tgt[p] : phys.qv_gpu[p],
                          frac; ndrange=(N, N, Nz))
            end

            for_panels_nosync() do p
                gpu.am[p] .*= am_to_mfx
                gpu.bm[p] .*= am_to_mfx
            end
            gchp_tracer_2d!(tracers, ws_vr.dp_work, gpu.am, gpu.bm,
                              cx_gpu, cy_gpu, gpu.xfx, gpu.yfx,
                              gc.area, rarea, grid, _ORD, ws_lr, ws_vr.m_save)
            for_panels_nosync() do p
                gpu.am[p] .*= inv_am_to_mfx
                gpu.bm[p] .*= inv_am_to_mfx
            end
        end

    else
        # Single step: use current dry dp (no next window available)
        if phys.qv_loaded[]
            for_panels_nosync() do p
                be = get_backend(ws_vr.dp_work[p])
                _compute_dry_dp_kernel!(be, 256)(ws_vr.dp_work[p], gpu.delp[p], phys.qv_gpu[p];
                          ndrange=(N, N, Nz))
            end
        else
            for_panels_nosync() do p; copyto!(ws_vr.dp_work[p], gpu.delp[p]); end
        end

        for_panels_nosync() do p
            gpu.am[p] .*= am_to_mfx
            gpu.bm[p] .*= am_to_mfx
        end
        gchp_tracer_2d!(tracers, ws_vr.dp_work, gpu.am, gpu.bm,
                          cx_gpu, cy_gpu, gpu.xfx, gpu.yfx,
                          gc.area, rarea, grid, _ORD, ws_lr, ws_vr.m_save)
        for_panels_nosync() do p
            gpu.am[p] .*= inv_am_to_mfx
            gpu.bm[p] .*= inv_am_to_mfx
        end
    end

    # ── Vertical remap (single, at end) — skipped when per_step_remap did it ──
    if !(has_next && phys.qv_loaded[] && per_step)
        for_panels_nosync() do p
            compute_air_mass_panel!(ws_vr.m_save[p], ws_vr.dp_work[p],
                                    gc.area[p], g_val, Nc, Nz, Hp)
        end
        compute_source_pe_from_evolved_mass!(ws_vr, ws_vr.m_save, gc, grid)

        if has_next
            ng = next_gpu(sched)
            if qv_tgt !== nothing
                compute_target_pressure_from_dry_delp_direct!(ws_vr, ng.delp,
                    qv_tgt, gc, grid)
            else
                compute_target_pressure_from_delp_direct!(ws_vr, ng.delp, gc, grid)
            end
        else
            compute_target_pressure_from_mass_direct!(ws_vr, ws_vr.m_save, gc, grid)
        end
        for (_, rm_t) in pairs(tracers)
            q_to_rm_panels!(rm_t, ws_vr.m_save, grid)
        end
        for (_, rm_t) in pairs(tracers)
            vertical_remap_cs!(rm_t, ws_vr.m_save, ws_vr, ws, gc, grid; hybrid_pe=true)
        end
        for (_, rm_t) in pairs(tracers)
            fillz_panels!(rm_t, ws_vr.dp_tgt, grid)
        end

        if has_next
            ng = next_gpu(sched)
            for (tname, rm_t) in pairs(tracers)
                scaling = gchp_calc_scaling_factor(rm_t, ws_vr.dp_tgt, ng.delp, gc, grid;
                    qv_panels=qv_tgt)
                @info "calcScaling: $tname = $scaling" maxlog=200
                apply_scaling_factor!(rm_t, scaling, grid)
            end
        end
    end

    # air.m = dry mass from prescribed endpoint
    if has_next
        ng = next_gpu(sched)
        if qv_tgt !== nothing
            for_panels_nosync() do p
                compute_air_mass_panel!(air.m[p], ng.delp[p], qv_tgt[p],
                                        gc.area[p], g_val, Nc, Nz, Hp)
            end
        else
            for_panels_nosync() do p
                compute_air_mass_panel!(air.m[p], ng.delp[p],
                                        gc.area[p], g_val, Nc, Nz, Hp)
            end
        end
    else
        for_panels_nosync() do p
            compute_air_mass_panel!(air.m[p], ws_vr.dp_work[p],
                                    gc.area[p], g_val, Nc, Nz, Hp)
        end
    end
    for_panels_nosync() do p; copyto!(air.m_ref[p], air.m[p]); end
    if phys.qv_loaded[]
        for_panels_nosync() do p
            air.m_wet[p] .= air.m[p] ./ max.(1 .- phys.qv_gpu[p], eps(FT))
        end
    else
        for_panels_nosync() do p; copyto!(air.m_wet[p], air.m[p]); end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# GCHP advection: moist-basis with interpolated QV
#
# Key differences from dry:
# 1. dp = moist DELP (no QV correction)
# 2. MFX = dry × /(1-QV) (moist mass flux)
# 3. Target PE = hybrid from evolved moist PS (column-preserving!)
# 4. Back-conversion uses QV_next from met data (NOT prognostic QV)
#    → no PPM/remap noise in QV, smooth met-data QV at end of window
# 5. air.m = dry mass from DELP_next × (1-QV_next)
#
# Since sum(DELP) = PS exactly in GEOS-IT, the moist basis is perfectly
# consistent with the hybrid ak/bk coefficients.
# ---------------------------------------------------------------------------
function _gchp_advection_moist!(tracers, sched, air, phys, model,
        grid::CubedSphereGrid{FT}, ws, ws_lr, ws_vr, gc,
        n_sub, dt_sub, step, _ORD, has_next) where FT
    gpu = current_gpu(sched)
    Nc, Nz, Hp = grid.Nc, grid.Nz, grid.Hp
    N = Nc + 2Hp
    cx_gpu, cy_gpu = gpu.cx, gpu.cy
    g_val = FT(grid.gravity)

    step[] += n_sub

    # ── Moist sub-step diagnostics: capture initial state (first window only) ──
    _do_diag = MOIST_DIAG[] !== nothing && !(MOIST_DIAG[]::MoistSubStepDiag{FT}).captured
    if _do_diag
        _diag = MOIST_DIAG[]::MoistSubStepDiag{FT}
        for p in 1:6
            _diag.qv_start[p]   .= Array(phys.qv_gpu[p])
            _diag.delp_start[p] .= Array(gpu.delp[p])
        end
        # q_dry_init: tracers are in rm form at this point (before rm_to_q)
        if haskey(tracers, :co2)
            for p in 1:6; _diag.q_dry_init[p] .= Array(tracers.co2[p]); end
        end
    end

    # Step 1: rm → q_wet (divide by moist air mass)
    for (_, rm_t) in pairs(tracers)
        rm_to_q_panels!(rm_t, air.m_wet, grid)
    end

    # Pre-compute constants
    mfx_dt = FT(model.met_data.mass_flux_dt)
    am_to_mfx = g_val * mfx_dt
    inv_am_to_mfx = FT(1) / am_to_mfx
    rarea = ntuple(p -> FT(1) ./ gc.area[p], 6)

    # ── n_sub loop with dp-reset (interpolated DELP, moist) ─────────────
    n_loop = has_next ? n_sub : 1
    per_step = model.advection_scheme.per_step_remap

    if has_next && per_step
        # ── Per-substep remap: matches GCHP offline_tracer_advection ────────
        ng = next_gpu(sched)

        # GCHP applies humidity correction ONCE before all substeps (GCHPctmEnv:1029).
        # Scale am/bm to Pa·m² and correct for humidity — keep corrected for all substeps.
        for_panels_nosync() do p
            gpu.am[p] .*= am_to_mfx
            gpu.bm[p] .*= am_to_mfx
        end
        for_panels_nosync() do p
            be = get_backend(gpu.am[p])
            _correct_mfx_humidity_kernel!(be, 256)(gpu.am[p], phys.qv_gpu[p], Hp, Nc; ndrange=(Nc+1, Nc, Nz))
            _correct_mfy_humidity_kernel!(be, 256)(gpu.bm[p], phys.qv_gpu[p], Hp, Nc; ndrange=(Nc, Nc+1, Nz))
        end

        # QV advection as tracer NQ+1 (GCHP: AdvCore:1068) was tested but provides
        # only ~3% correction to the moist path artifact (see MoistSubStepDiag analysis).
        # The fundamental issue is that PPM homogenizes q_total, erasing the QV imprint,
        # while QV retains its structure — so back-conversion reintroduces QV patterns.
        # Disabled to avoid GPU allocation overhead (6 CuArrays per window).
        tracers_with_qv = tracers

        for _isub in 1:n_loop
            frac_start = FT(_isub - 1) / FT(n_loop)
            frac_end   = FT(_isub)     / FT(n_loop)

            # Reset dp_work to interpolated moist DELP at start of substep
            for_panels_nosync() do p
                be = get_backend(ws_vr.dp_work[p])
                _interpolate_dp_kernel!(be, 256)(ws_vr.dp_work[p], gpu.delp[p], ng.delp[p],
                          frac_start; ndrange=(N, N, Nz))
            end

            # Horizontal advection (fluxes already corrected for humidity)
            gchp_tracer_2d!(tracers_with_qv, ws_vr.dp_work, gpu.am, gpu.bm,
                              cx_gpu, cy_gpu, gpu.xfx, gpu.yfx,
                              gc.area, rarea, grid, _ORD, ws_lr, ws_vr.m_save)

            # ── Diag: capture q_wet after horizontal advection (substep 1, first window) ──
            if _do_diag && _isub == 1 && haskey(tracers, :co2)
                for p in 1:6
                    (MOIST_DIAG[]::MoistSubStepDiag{FT}).q_wet_post_hadv[p] .= Array(tracers.co2[p])
                end
            end

            # Source PE from evolved dpA (moist)
            for_panels_nosync() do p
                compute_air_mass_panel!(ws_vr.m_save[p], ws_vr.dp_work[p],
                                        gc.area[p], g_val, Nc, Nz, Hp)
            end
            compute_source_pe_from_evolved_mass!(ws_vr, ws_vr.m_save, gc, grid)

            # Target PE: direct cumsum from prescribed moist DELP, with column
            # mass scaled proportionally to match evolved surface pressure.
            # (Same approach as dry fix7 — hybrid PE accumulates drift over 384 substeps.)
            if _isub < n_loop
                for_panels_nosync() do p
                    be = get_backend(ws_vr.dp_work[p])
                    _interpolate_dp_kernel!(be, 256)(ws_vr.dp_work[p], gpu.delp[p], ng.delp[p],
                              frac_end; ndrange=(N, N, Nz))
                end
                compute_target_pressure_from_delp_direct!(ws_vr, ws_vr.dp_work, gc, grid)
                for_panels_nosync() do p
                    be = get_backend(ws_vr.pe_tgt[p])
                    _scale_dp_tgt_to_source_ps_kernel!(be, 256)(
                        ws_vr.pe_tgt[p], ws_vr.dp_tgt[p],
                        ws_vr.pe_src[p], ws_vr.ps_src[p],
                        Nc, Nz; ndrange=(Nc, Nc))
                end
            else
                compute_target_pe_from_evolved_ps!(ws_vr, gc, grid)
            end

            # q → rm (moist), remap, fillz — including QV tracer
            for (_, rm_t) in pairs(tracers_with_qv)
                q_to_rm_panels!(rm_t, ws_vr.m_save, grid)
            end
            for (_, rm_t) in pairs(tracers_with_qv)
                vertical_remap_cs!(rm_t, ws_vr.m_save, ws_vr, ws, gc, grid; hybrid_pe=true)
            end
            for (_, rm_t) in pairs(tracers_with_qv)
                fillz_panels!(rm_t, ws_vr.dp_tgt, grid)
            end

            # ── Diag: capture rm after vertical remap + fillz (substep 1, first window) ──
            if _do_diag && _isub == 1 && haskey(tracers, :co2)
                for p in 1:6
                    (MOIST_DIAG[]::MoistSubStepDiag{FT}).rm_post_vremap[p] .= Array(tracers.co2[p])
                end
            end

            if _isub < n_loop
                # No intermediate scaling — proportional dp_tgt scaling + surface-locked
                # PE ensures column mass conservation.

                # Convert rm → q using TARGET mass (dp_tgt, surface-locked).
                for_panels_nosync() do p
                    be = get_backend(ws_vr.dp_work[p])
                    _copy_dp_tgt_to_dp_work_kernel!(be, 256)(
                        ws_vr.dp_work[p], ws_vr.dp_tgt[p], Hp, Nc, Nz;
                        ndrange=(Nc, Nc, Nz))
                end
                for_panels_nosync() do p
                    compute_air_mass_panel!(ws_vr.m_save[p], ws_vr.dp_work[p],
                                            gc.area[p], g_val, Nc, Nz, Hp)
                end
                for (_, rm_t) in pairs(tracers_with_qv)
                    rm_to_q_panels!(rm_t, ws_vr.m_save, grid)
                end
            else
                # Last substep: scale against actual moist ng.delp (all tracers + QV)
                for (tname, rm_t) in pairs(tracers_with_qv)
                    scaling = gchp_calc_scaling_factor(rm_t, ws_vr.dp_tgt, ng.delp, gc, grid)
                    @info "calcScaling: $tname = $scaling" maxlog=200
                    apply_scaling_factor!(rm_t, scaling, grid)
                end
                # tracers remain in rm form for back-conversion below
            end
        end

        # Restore am/bm: reverse humidity correction + unscale
        for_panels_nosync() do p
            be = get_backend(gpu.am[p])
            _reverse_mfx_humidity_kernel!(be, 256)(gpu.am[p], phys.qv_gpu[p], Hp, Nc; ndrange=(Nc+1, Nc, Nz))
            _reverse_mfy_humidity_kernel!(be, 256)(gpu.bm[p], phys.qv_gpu[p], Hp, Nc; ndrange=(Nc, Nc+1, Nz))
        end
        for_panels_nosync() do p
            gpu.am[p] .*= inv_am_to_mfx
            gpu.bm[p] .*= inv_am_to_mfx
        end

    elseif has_next
        ng = next_gpu(sched)
        for _isub in 1:n_loop
            frac = FT(_isub - 1) / FT(n_loop)
            for_panels_nosync() do p
                be = get_backend(ws_vr.dp_work[p])
                _interpolate_dp_kernel!(be, 256)(ws_vr.dp_work[p], gpu.delp[p], ng.delp[p],
                          frac; ndrange=(N, N, Nz))
            end

            for_panels_nosync() do p
                gpu.am[p] .*= am_to_mfx
                gpu.bm[p] .*= am_to_mfx
            end
            for_panels_nosync() do p
                be = get_backend(gpu.am[p])
                _correct_mfx_humidity_kernel!(be, 256)(gpu.am[p], phys.qv_gpu[p], Hp, Nc; ndrange=(Nc+1, Nc, Nz))
                _correct_mfy_humidity_kernel!(be, 256)(gpu.bm[p], phys.qv_gpu[p], Hp, Nc; ndrange=(Nc, Nc+1, Nz))
            end
            gchp_tracer_2d!(tracers, ws_vr.dp_work, gpu.am, gpu.bm,
                              cx_gpu, cy_gpu, gpu.xfx, gpu.yfx,
                              gc.area, rarea, grid, _ORD, ws_lr, ws_vr.m_save)
            for_panels_nosync() do p
                be = get_backend(gpu.am[p])
                _reverse_mfx_humidity_kernel!(be, 256)(gpu.am[p], phys.qv_gpu[p], Hp, Nc; ndrange=(Nc+1, Nc, Nz))
                _reverse_mfy_humidity_kernel!(be, 256)(gpu.bm[p], phys.qv_gpu[p], Hp, Nc; ndrange=(Nc, Nc+1, Nz))
            end
            for_panels_nosync() do p
                gpu.am[p] .*= inv_am_to_mfx
                gpu.bm[p] .*= inv_am_to_mfx
            end
        end
    else
        for_panels_nosync() do p; copyto!(ws_vr.dp_work[p], gpu.delp[p]); end
        for_panels_nosync() do p
            gpu.am[p] .*= am_to_mfx
            gpu.bm[p] .*= am_to_mfx
        end
        for_panels_nosync() do p
            be = get_backend(gpu.am[p])
            _correct_mfx_humidity_kernel!(be, 256)(gpu.am[p], phys.qv_gpu[p], Hp, Nc; ndrange=(Nc+1, Nc, Nz))
            _correct_mfy_humidity_kernel!(be, 256)(gpu.bm[p], phys.qv_gpu[p], Hp, Nc; ndrange=(Nc, Nc+1, Nz))
        end
        gchp_tracer_2d!(tracers, ws_vr.dp_work, gpu.am, gpu.bm,
                          cx_gpu, cy_gpu, gpu.xfx, gpu.yfx,
                          gc.area, rarea, grid, _ORD, ws_lr, ws_vr.m_save)
        for_panels_nosync() do p
            be = get_backend(gpu.am[p])
            _reverse_mfx_humidity_kernel!(be, 256)(gpu.am[p], phys.qv_gpu[p], Hp, Nc; ndrange=(Nc+1, Nc, Nz))
            _reverse_mfy_humidity_kernel!(be, 256)(gpu.bm[p], phys.qv_gpu[p], Hp, Nc; ndrange=(Nc, Nc+1, Nz))
        end
        for_panels_nosync() do p
            gpu.am[p] .*= inv_am_to_mfx
            gpu.bm[p] .*= inv_am_to_mfx
        end
    end

    # ── Vertical remap (single, at end) — skipped when per_step_remap did it ──
    if !(has_next && per_step)
        for_panels_nosync() do p
            compute_air_mass_panel!(ws_vr.m_save[p], ws_vr.dp_work[p],
                                    gc.area[p], g_val, Nc, Nz, Hp)
        end
        compute_source_pe_from_evolved_mass!(ws_vr, ws_vr.m_save, gc, grid)
        compute_target_pe_from_evolved_ps!(ws_vr, gc, grid)

        for (_, rm_t) in pairs(tracers)
            q_to_rm_panels!(rm_t, ws_vr.m_save, grid)
        end
        for (_, rm_t) in pairs(tracers)
            vertical_remap_cs!(rm_t, ws_vr.m_save, ws_vr, ws, gc, grid; hybrid_pe=true)
        end
        for (_, rm_t) in pairs(tracers)
            fillz_panels!(rm_t, ws_vr.dp_tgt, grid)
        end

        if has_next
            ng = next_gpu(sched)
            for (tname, rm_t) in pairs(tracers)
                scaling = gchp_calc_scaling_factor(rm_t, ws_vr.dp_tgt, ng.delp, gc, grid)
                @info "calcScaling: $tname = $scaling" maxlog=200
                apply_scaling_factor!(rm_t, scaling, grid)
            end
        end
    end

    # Back-conversion: wet→dry using met-data QV.
    # Note: GCHP uses advected QV (tracer NQ+1) but testing showed only ~3% improvement
    # due to PPM homogenizing q_total while QV retains its structure.
    qv_back = phys.qv_next_loaded[] ? phys.qv_next_gpu : phys.qv_gpu

    # ── Diag: capture delp_end (first window only) ──
    if _do_diag && has_next
        _ng = next_gpu(sched)
        _diag = MOIST_DIAG[]::MoistSubStepDiag{FT}
        for p in 1:6; _diag.delp_end[p] .= Array(_ng.delp[p]); end
    end

    if has_next
        ng = next_gpu(sched)
        for_panels_nosync() do p
            compute_air_mass_panel!(ws_vr.m_save[p], ng.delp[p],
                                    gc.area[p], g_val, Nc, Nz, Hp)
        end
    else
        for_panels_nosync() do p
            compute_air_mass_panel!(ws_vr.m_save[p], ws_vr.dp_work[p],
                                    gc.area[p], g_val, Nc, Nz, Hp)
        end
    end

    for (_, rm_t) in pairs(tracers)
        rm_to_q_panels!(rm_t, ws_vr.m_save, grid)
    end
    # ── Diag: capture qv_back AFTER rm_to_q conversion ──
    if _do_diag
        _diag = MOIST_DIAG[]::MoistSubStepDiag{FT}
        for p in 1:6
            _diag.qv_back[p] .= Array(qv_back[p])
        end
    end

    for (_, q_t) in pairs(tracers)
        for_panels_nosync() do p
            be = get_backend(q_t[p])
            _divide_by_1_minus_qv_kernel!(be, 256)(q_t[p], qv_back[p], Hp; ndrange=(Nc, Nc, Nz))
        end
    end

    # ── Diag: capture q_dry_final after wet→dry back-conversion (first window) ──
    if _do_diag && haskey(tracers, :co2)
        for p in 1:6
            (MOIST_DIAG[]::MoistSubStepDiag{FT}).q_dry_final[p] .= Array(tracers.co2[p])
        end
        (MOIST_DIAG[]::MoistSubStepDiag{FT}).captured = true
        @info "MoistSubStepDiag: first window captured"
    end

    if has_next
        ng = next_gpu(sched)
        for_panels_nosync() do p
            compute_air_mass_panel!(air.m[p], ng.delp[p], qv_back[p],
                                    gc.area[p], g_val, Nc, Nz, Hp)
        end
    else
        for_panels_nosync() do p
            compute_air_mass_panel!(air.m[p], ws_vr.dp_work[p],
                                    gc.area[p], g_val, Nc, Nz, Hp)
        end
    end

    for (_, q_t) in pairs(tracers)
        q_to_rm_panels!(q_t, air.m, grid)
    end

    for_panels_nosync() do p; copyto!(air.m_ref[p], air.m[p]); end
    for_panels_nosync() do p
        air.m_wet[p] .= air.m[p] ./ max.(1 .- qv_back[p], eps(FT))
    end
    return nothing
end

# ---------------------------------------------------------------------------

function advection_phase!(tracers, sched, air, phys, model,
                           grid::CubedSphereGrid{FT},
                           ws, n_sub, dt_sub, step;
                           ws_lr=nothing, ws_vr=nothing, gc=nothing,
                           geom_gchp=nothing, ws_gchp=nothing,
                           has_next::Bool=false) where FT
    gpu = current_gpu(sched)
    Nc, Nz, Hp = grid.Nc, grid.Nz, grid.Hp
    # GCHP-faithful: get CX/CY from GPU met buffer
    cx_gpu = gpu.cx
    cy_gpu = gpu.cy

    _use_gchp = _needs_gchp(model.advection_scheme)
    _ORD = model.advection_scheme isa PPMAdvection ?
        Val(_ppm_order(model.advection_scheme)) : Val(4)

    if _use_gchp && ws_vr !== nothing && cx_gpu !== nothing && gpu.xfx !== nothing
        # ═══════════════════════════════════════════════════════════════════
        # GCHP PATH: dispatch on transport basis (dry or moist)
        # ═══════════════════════════════════════════════════════════════════
        _basis = get(model.metadata, "pressure_basis", "dry")
        if _basis == "moist" && phys.qv_loaded[]
            _gchp_advection_moist!(tracers, sched, air, phys, model, grid,
                ws, ws_lr, ws_vr, gc, n_sub, dt_sub, step, _ORD, has_next)
        else
            _gchp_advection_dry!(tracers, sched, air, phys, model, grid,
                ws, ws_lr, ws_vr, gc, n_sub, dt_sub, step, _ORD, has_next)
        end

        # ── Convection AFTER full advection ─────────────────────────
        if phys.cmfmc_loaded[]
            dt_conv = FT(n_sub) * dt_sub
            for (_, rm_t) in pairs(tracers)
                convect!(rm_t, air.m_wet, phys.cmfmc_gpu, gpu.delp,
                          model.convection, grid, dt_conv, phys.planet;
                          dtrain_panels=phys.dtrain_loaded[] ? phys.dtrain_gpu : nothing,
                          workspace=phys.ras_workspace)
            end
        end

    elseif ws_vr !== nothing
        # ═══════════════════════════════════════════════════════════════════
        # REMAP PATH: Standard Lin-Rood + vertical remap (prescale/rescale)
        # ═══════════════════════════════════════════════════════════════════

        # Save prescribed m (from compute_air_mass_phase!)
        for_panels_nosync() do p; copyto!(ws_vr.m_save[p], air.m[p]); end

        for _ in 1:n_sub
            step[] += 1

            for (_, rm_t) in pairs(tracers)
                for_panels_nosync() do p; copyto!(air.m[p], ws_vr.m_save[p]); end

                fv_tp_2d_cs!(rm_t, air.m, gpu.am, gpu.bm,
                              grid, _ORD, ws, ws_lr;
                              damp_coeff=model.advection_scheme.damp_coeff)
                fv_tp_2d_cs!(rm_t, air.m, gpu.am, gpu.bm,
                              grid, _ORD, ws, ws_lr;
                              damp_coeff=FT(0))

                for_panels_nosync() do p
                    rm_t[p] .*= ws_vr.m_save[p] ./ air.m[p]
                end
            end
        end

        # ── Convective transport ONCE per window (after all substeps) ─────
        # Both TM5 and GCHP apply convection as a separate operator outside
        # the advection substep loop. Interleaving convection with substeps
        # causes 12× over-mixing because RAS recomputes q_cloud from the
        # already-mixed environment at each call (nonlinear feedback).
        # RAS internal subcycling handles CFL stability for the larger dt.
        for_panels_nosync() do p; copyto!(air.m[p], ws_vr.m_save[p]); end
        if phys.cmfmc_loaded[]
            dt_conv = FT(n_sub) * dt_sub
            for (_, rm_t) in pairs(tracers)
                convect!(rm_t, air.m_wet, phys.cmfmc_gpu, gpu.delp,
                          model.convection, grid, dt_conv, phys.planet;
                          dtrain_panels=phys.dtrain_loaded[] ? phys.dtrain_gpu : nothing,
                          workspace=phys.ras_workspace)
            end
        end

        # GCHP-style PE computation (fv_tracer2d.F90:988-1035):
        # Source PE: direct cumsum from actual air mass (m_save = DELP×(1-QV)×area/g).
        #   Derived inside remap kernel from m_src (hybrid_pe=false).
        #   GCHP uses cumsum of post-horizontal dpA — our m_save is equivalent
        #   (prescribed dry air mass on current window's pressure structure).
        # Target PE: direct cumsum of next-window dry DELP (no hybrid reconstruction).
        #   The hybrid formula (PE=ak+bk×PS) deviates from actual met DELP by
        #   0.1-1% per level (up to 250 Pa), causing systematic vertical pumping.
        #   GCHP compensates with calcScalingFactor; direct cumsum avoids the issue.
        if has_next
            ng = next_gpu(sched)
            if phys.qv_loaded[]
                compute_target_pressure_from_dry_delp_direct!(ws_vr, ng.delp,
                    phys.qv_gpu, gc, grid)
            else
                compute_target_pressure_from_delp_direct!(ws_vr, ng.delp, gc, grid)
            end
        else
            compute_target_pressure_from_mass_direct!(ws_vr, air.m, gc, grid)
        end

        # Vertical remap: source PE from m_src inside kernel (hybrid_pe=false)
        for (_, rm_t) in pairs(tracers)
            vertical_remap_cs!(rm_t, ws_vr.m_save, ws_vr, ws, gc, grid;
                               hybrid_pe=false)
        end

        # Update air.m to target state (m = dp_tgt * area / g)
        update_air_mass_from_target!(air.m, ws_vr, gc, grid)
        # Copy to m_ref for output/diagnostics
        for_panels_nosync() do p; copyto!(air.m_ref[p], air.m[p]); end

        # Recompute m_wet from updated air.m for post-advection physics.
        # After remap, air.m is on target pressure basis; m_wet must match
        # so that diffusion's q = rm/m_wet is consistent with remapped rm.
        # m_wet = m_dry / (1-qv). Uses current-window QV (<0.1% approx).
        if phys.qv_loaded[]
            for_panels_nosync() do p
                air.m_wet[p] .= air.m[p] ./ max.(1 .- phys.qv_gpu[p], eps(FT))
            end
        else
            for_panels_nosync() do p; copyto!(air.m_wet[p], air.m[p]); end
        end
    else
        # ═══════════════════════════════════════════════════════════════════
        # STRANG PATH: Existing cm-based Z-advection
        # ═══════════════════════════════════════════════════════════════════
        for _ in 1:n_sub
            step[] += 1

            # Advect each tracer independently (m reset per tracer)
            # For Prather, get per-tracer CS workspace (lazy-allocated)
            _cs_pw_dict = model.advection_scheme isa PratherAdvection ?
                _get_cs_prather_ws(tracers, grid, model.architecture) : nothing
            for (tname, rm_t) in pairs(tracers)
                for_panels_nosync() do p
                    copyto!(air.m[p], air.m_ref[p])
                end
                _pw_cs = _cs_pw_dict !== nothing ? _cs_pw_dict[tname] : nothing
                _apply_advection_cs!(rm_t, air.m, gpu.am, gpu.bm, gpu.cm,
                                      grid, model.advection_scheme, ws;
                                      ws_lr, cx=cx_gpu, cy=cy_gpu,
                                      geom_gchp, ws_gchp, pw_cs=_pw_cs)
            end
            # air.m now holds m_evolved (same for all tracers)

            # Per-cell mass fixer: rm = (rm / m_evolved) × m_ref
            if get(model.metadata, "mass_fixer", true)
                for (_, rm_t) in pairs(tracers)
                    for_panels_nosync() do p
                        apply_mass_fixer!(rm_t[p], air.m_ref[p], air.m[p], Nc, Nz, Hp)
                    end
                end
            end
        end

        # ── Convective transport ONCE per window (after all substeps) ─────
        # See remap path comment: TM5/GCHP apply convection outside substeps.
        if phys.cmfmc_loaded[]
            dt_conv = FT(n_sub) * dt_sub
            for (_, rm_t) in pairs(tracers)
                convect!(rm_t, air.m_wet, phys.cmfmc_gpu, gpu.delp,
                          model.convection, grid, dt_conv, phys.planet;
                          dtrain_panels=phys.dtrain_loaded[] ? phys.dtrain_gpu : nothing,
                          workspace=phys.ras_workspace)
            end
        end
    end
end

# =====================================================================
# Phase 9: Post-advection physics (BLD + PBL diffusion + chemistry)
# =====================================================================

"""Post-advection physics: BLD diffusion, PBL diffusion, chemistry.
LL tracers are rm — convert to VMR for operators, then back (same m_ref → exact).
TM5-faithful: moist basis for boundary conversions."""
function post_advection_physics!(tracers, sched, air, phys, model,
                                  grid::LatitudeLongitudeGrid, dt_window, dw)
    gpu = current_gpu(sched)

    # rm → c (moist VMR) for diffusion and chemistry
    for (_, rm) in pairs(tracers)
        rm ./= gpu.m_ref
    end

    _apply_bld!(tracers, dw)

    if phys.sfc_loaded[]
        diffuse_pbl!(tracers, gpu.Δp,
                      phys.pbl_sfc_gpu.pblh, phys.pbl_sfc_gpu.ustar,
                      phys.pbl_sfc_gpu.hflux, phys.pbl_sfc_gpu.t2m,
                      phys.w_scratch,
                      model.diffusion, grid, dt_window, phys.planet)
    end

    apply_chemistry!(tracers, grid, model.chemistry, dt_window)

    # c → rm
    for (_, c) in pairs(tracers)
        c .*= gpu.m_ref
    end
end

function post_advection_physics!(tracers, sched, air, phys, model,
                                  grid::CubedSphereGrid, dt_window, dw)
    gpu = current_gpu(sched)
    Nc, Nz, Hp = grid.Nc, grid.Nz, grid.Hp

    # Use DRY air mass for rm↔q conversion in diffusion. GeosChem's
    # pbl_mix_mod.F90 uses AD = State_Met%AD (dry air mass) for PBL mixing.
    # Using moist mass creates QV-dependent VMR artifacts: ~0.8 ppm
    # tropical-polar bias in dry VMR output from q_wet/(1-QV) conversion.
    # TODO: verify convection should also switch to dry (GCHP convection_mod
    # uses BMASS = DELP_DRY for RAS, but CMFMC flux is moist — needs care).
    for (_, rm_t) in pairs(tracers)
        _apply_bld_cs!(rm_t, air.m, dw, Nc, Nz, Hp)
    end

    if phys.sfc_loaded[] && phys.has_pbl
        for (_, rm_t) in pairs(tracers)
            diffuse_pbl!(rm_t, air.m, gpu.delp,
                          phys.pbl_sfc_gpu.pblh, phys.pbl_sfc_gpu.ustar,
                          phys.pbl_sfc_gpu.hflux, phys.pbl_sfc_gpu.t2m,
                          phys.w_scratch,
                          model.diffusion, grid, dt_window, phys.planet)
        end
    end

    apply_chemistry!(tracers, grid, model.chemistry, dt_window)
end

# =====================================================================
# Phase 10: Global mass correction (CS only)
# =====================================================================

"""Apply global mass fixer. No-op for LL; scales tracers for CS."""
apply_mass_correction!(tracers, grid::LatitudeLongitudeGrid, diag;
                        mass_fixer::Bool=true, mass_fixer_tracers::Vector{String}=String[]) = nothing

function apply_mass_correction!(tracers, grid::CubedSphereGrid, diag;
                                 mass_fixer::Bool=true,
                                 mass_fixer_tracers::Vector{String}=String[])
    mass_fixer || return
    isempty(diag.pre_adv_mass) && return
    diag.fix_value = apply_global_mass_fixer!(tracers, grid, diag.pre_adv_mass;
                                               fix_interval_scale=diag.fix_interval_scale,
                                               fix_total_scale=diag.fix_total_scale,
                                               allowed_tracers=mass_fixer_tracers)
end

# =====================================================================
# Phase 11: Compute output air mass (dry correction)
# =====================================================================

"""Compute air mass for output. If QV loaded and compatible, returns dry mass."""
function compute_output_mass(sched, air, phys, grid::LatitudeLongitudeGrid)
    gpu = current_gpu(sched)
    if phys.qv_loaded[] && size(phys.qv_gpu) == size(gpu.m_ref)
        phys.m_dry .= gpu.m_ref .* (1 .- phys.qv_gpu)
        return phys.m_dry
    end
    return gpu.m_ref
end

"""Convert rm tracers to dry VMR for output. LL: c_dry = rm / m_dry.
Creates GPU temporaries (once per output interval). CS: tracers are already rm."""
rm_to_vmr(tracers, sched, phys, grid::LatitudeLongitudeGrid) =
    map(rm -> rm ./ ll_dry_mass(phys), tracers)
rm_to_vmr(tracers, sched, phys, grid::CubedSphereGrid) = tracers

function compute_output_mass(sched, air, phys, grid::CubedSphereGrid)
    # Use current air.m — after vertical remap, air.m is the target mass
    # and rm has been remapped to match. VMR = rm / air.m is correct.
    # (air.m_ref is the START-of-window mass, inconsistent with post-remap rm.)
    return air.m
end

# =====================================================================
# Phase 12: Build met_fields for output writers
# =====================================================================

"""Build met_fields NamedTuple for output writers."""
function build_met_fields(sched, phys, grid::LatitudeLongitudeGrid, half_dt, dt_window)
    gpu = current_gpu(sched)
    base = (; ps=Array(gpu.ps))
    if phys.sfc_loaded[] && phys.has_pbl
        base = merge(base, (; pblh=Array(phys.pbl_sfc_gpu.pblh)))
    end
    return base
end

function build_met_fields(sched, phys, grid::CubedSphereGrid, half_dt, dt_window)
    gpu = current_gpu(sched)
    base = (; ps=phys.ps_cpu,
              mass_flux_x=gpu.am, mass_flux_y=gpu.bm,
              mf_scale=half_dt, dt_window=dt_window)

    if phys.sfc_loaded[] && phys.has_pbl && phys.troph_loaded[]
        base = merge(base, (; pblh=phys.pbl_sfc_cpu.pblh, troph=phys.troph_cpu))
    elseif phys.sfc_loaded[] && phys.has_pbl
        base = merge(base, (; pblh=phys.pbl_sfc_cpu.pblh))
    elseif phys.troph_loaded[]
        base = merge(base, (; troph=phys.troph_cpu))
    end

    if phys.qv_loaded[]
        base = merge(base, (; qv=phys.qv_cpu))
    end
    return base
end

# =====================================================================
# IC output (first window only)
# =====================================================================

"""Write IC output snapshot (no-op for LL, writes t=0 output for CS)."""
write_ic_output!(writers, model, tracers, sched, air, phys, gc,
                  grid::LatitudeLongitudeGrid, half_dt, dt_window) = nothing

function write_ic_output!(writers, model, tracers, sched, air, phys, gc,
                            grid::CubedSphereGrid, half_dt, dt_window)
    gpu = current_gpu(sched)

    # Build IC met_fields
    met_ic = (; ps=phys.ps_cpu,
                mass_flux_x=gpu.am, mass_flux_y=gpu.bm,
                mf_scale=half_dt, dt_window=dt_window)
    if phys.sfc_loaded[] && phys.has_pbl
        met_ic = merge(met_ic, (; pblh=phys.pbl_sfc_cpu.pblh))
    end
    if phys.troph_loaded[]
        met_ic = merge(met_ic, (; troph=phys.troph_cpu))
    end

    # IC mass (dry if QV available)
    ic_mass = compute_output_mass(sched, air, phys, grid)

    for writer in writers
        write_output!(writer, model, 0.0;
                      air_mass=ic_mass, tracers=tracers, met_fields=met_ic)
    end
end

# =====================================================================
# Progress bar update (unified format)
# =====================================================================

"""Update progress bar with timing + diagnostics."""
function update_progress!(prog, diag, grid::LatitudeLongitudeGrid,
                           w, step, dt_sub, wall_start, t_io, t_gpu, t_out)
    sim_time = Float64(step * dt_sub)
    sv = Pair{Symbol,Any}[
        :day  => @sprintf("%.1f", sim_time / 86400),
        :rate => @sprintf("%.2f s/win", w > 1 ? (time() - wall_start) / w : 0.0)]
    isempty(diag.showvalue) || push!(sv, :mass => diag.showvalue)
    next!(prog; showvalues=sv)
end

function update_progress!(prog, diag, grid::CubedSphereGrid,
                           w, step, dt_sub, wall_start, t_io, t_gpu, t_out)
    sv = Pair{Symbol,Any}[
        :day  => div(w - 1, 24) + 1,
        :IO   => @sprintf("%.2f s/win", t_io / w),
        :GPU  => @sprintf("%.2f s/win", t_gpu / w),
        :Out  => @sprintf("%.2f s/win", t_out / w)]
    isempty(diag.cfl_value)  || push!(sv, :CFL  => diag.cfl_value)
    isempty(diag.fix_value)  || push!(sv, :fix  => diag.fix_value)
    isempty(diag.showvalue)  || push!(sv, :mass => diag.showvalue)
    next!(prog; showvalues=sv)
end

# =====================================================================
# Finalize simulation (summary logging + output finalization)
# =====================================================================

"""Finalize simulation: log mass conservation summary + close output writers."""
function finalize_simulation!(writers, diag, tracers, grid,
                                wall_start, n_win, step, t_io, t_gpu, t_out;
                                t_phases=nothing)
    wall_total = time() - wall_start

    # Final mass summary (Δ = mass closure bias vs expected = initial + emissions)
    if !isempty(diag.expected_mass)
        final_mass = compute_mass_totals(tracers, grid)
        for tname in sort(collect(keys(final_mass)))
            total = final_mass[tname]
            if haskey(diag.expected_mass, tname) && diag.expected_mass[tname] != 0.0
                rel = (total - diag.expected_mass[tname]) /
                      abs(diag.expected_mass[tname]) * 100
                @info @sprintf("  Final mass %s: %.6e kg (Δ=%.4e%%)", tname, total, rel)
            else
                @info @sprintf("  Final mass %s: %.6e kg", tname, total)
            end
        end
    end

    # Timing summary
    if t_io > 0 || t_gpu > 0
        @info @sprintf(
            "Simulation complete: %d steps, %.1fs | avg IO=%.2f GPU=%.2f Out=%.2f s/win",
            step, wall_total, t_io / n_win, t_gpu / n_win, t_out / n_win)
    else
        dt_per_win = n_win > 0 ? wall_total / n_win : 0.0
        @info @sprintf("Simulation complete: %d steps, %.1fs (%.2fs/win)",
                       step, wall_total, dt_per_win)
    end

    for writer in writers
        finalize_output!(writer)
    end
end
