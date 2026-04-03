# ---------------------------------------------------------------------------
# TM5 matrix convection scheme — forward
#
# Faithful port of TM5's convection algorithm from:
#   deps/tm5/base/src/tm5_conv.F90      (TM5_Conv_Matrix, TM5_Conv_Apply)
#   deps/tm5/base/src/convection.F90    (driver)
#   deps/tm5/base/src/tmphys_convec.F90 (ConvCloudDim)
#
# Key differences from our TiedtkeConvection:
#   - Uses 4 separate fields (entu, detu, entd, detd) not 1 (CMFMC)
#   - Builds a full Nz×Nz transfer matrix per column
#   - Applies via implicit LU solve (unconditionally stable)
#   - Tracks updraft/downdraft separately with retention fractions
#
# Level convention:
#   TM5 internal: bottom-to-top (k=1=surface, k=lm=TOA)
#   Our data:     top-to-bottom (k=1=TOA, k=Nz=surface)
#   We reverse columns at the interface to use TM5's algorithm verbatim.
#
# The implicit scheme solves:  conv1 * rm_new = rm_old
# where conv1 = I - dt * D  (D = flux divergence operator)
# This is backward Euler: rm_new = rm_old + dt * D * rm_new
#
# GPU strategy:
#   One KA kernel per column does: (1) build transfer matrix, (2) Gaussian
#   elimination without pivoting, (3) forward/back substitution per tracer.
#   The lmax x lmax matrix is stored in a pre-allocated 4D workspace array
#   conv1_ws[row, col, i, j]. The matrix is diagonal-dominant so pivoting
#   is not needed.
#
#   For CUDA, a faster path using cublasSgetrfBatched/cublasSgetrsBatched
#   could be added in the CUDA extension. The KA kernel approach works on
#   all backends (CPU, CUDA, Metal).
# ---------------------------------------------------------------------------

using ..Fields: interior, AbstractField
using ..Grids: grid_size, floattype, LatitudeLongitudeGrid, CubedSphereGrid
using ..Parameters: PlanetParameters
using LinearAlgebra: lu!, ldiv!
using KernelAbstractions: @kernel, @index, synchronize, get_backend

"""
Return the modifiable 3D array for a tracer (Field or raw array).
"""
_tm5_tracer_data(t) = t isa AbstractField ? interior(t) : t

# =====================================================================
# Pure matrix builder — TM5 level convention (k=1=surface, k=lm=TOA)
#
# Faithful port of TM5_Conv_Matrix from tm5_conv.F90 lines 32-186.
# =====================================================================

"""
    tm5_conv_matrix!(conv1, m, entu, detu, entd, detd, lmx, li, ld, dt)

Build the TM5 convection transfer matrix for a single column.
All input arrays use TM5's bottom-to-top convention (k=1=surface).
Returns `lmc::Int`: highest active convection level (0 = no convection).
"""
function tm5_conv_matrix!(conv1::AbstractMatrix{FT},
                          m::AbstractVector{FT},
                          entu::AbstractVector{FT}, detu::AbstractVector{FT},
                          entd::AbstractVector{FT}, detd::AbstractVector{FT},
                          lmx::Int, li::Int, ld::Int, dt::FT) where FT

    f  = zeros(FT, lmx + 1, lmx)
    fu = zeros(FT, lmx + 1, lmx)
    amu = zeros(FT, lmx + 1)
    amd = zeros(FT, lmx + 1)

    @inbounds for k in 1:li
        amu[k+1] = amu[k] + entu[k] - detu[k]
        if amu[k+1] > zero(FT)
            zxi = max(zero(FT), one(FT) - detu[k] / (amu[k] + entu[k]))
        else
            amu[k+1] = zero(FT)
            zxi = zero(FT)
        end
        for kk in 1:k-1
            fu[k+1, kk] = fu[k, kk] * zxi
        end
        fu[k+1, k] = entu[k] / m[k] * zxi
    end

    @inbounds for k in ld:-1:2
        amd[k] = amd[k+1] - entd[k] + detd[k]
        if amd[k] < zero(FT)
            zxi = max(zero(FT), one(FT) + detd[k] / (amd[k+1] - entd[k]))
        else
            amd[k] = zero(FT)
            zxi = zero(FT)
        end
        for kk in k+1:ld
            f[k, kk] = f[k+1, kk] * zxi
        end
        f[k, k] = -entd[k] / m[k] * zxi
    end

    @inbounds for k in 1:lmx-1
        for kk in 1:lmx
            f[k+1, kk] = fu[k+1, kk] + f[k+1, kk]
        end
        f[k+1, k+1] = f[k+1, k+1] - amu[k+1] / m[k+1]
        f[k+1, k] = f[k+1, k] - amd[k+1] / m[k]
    end

    lmc = 0
    fill!(conv1, zero(FT))
    @inbounds for k in 1:lmx
        for kk in 1:lmx
            fk_below = f[k, kk]
            fk_above = f[k+1, kk]
            if fk_below != fk_above
                conv1[k, kk] = -dt * (fk_below - fk_above)
                lmc = max(lmc, max(k, kk))
            end
        end
        conv1[k, k] += one(FT)
    end

    return lmc
end

# =====================================================================
# Cloud dimension diagnostics — port of ConvCloudDim
# =====================================================================

function _conv_cloud_dim(detu::AbstractVector{FT}, entd::AbstractVector{FT},
                         lmx::Int) where FT
    li = 0
    @inbounds for k in lmx:-1:1
        if detu[k] > zero(FT); li = k; break; end
    end
    ld = 0
    @inbounds for k in lmx:-1:1
        if entd[k] > zero(FT); ld = k; break; end
    end
    return li, ld
end

# =====================================================================
# GPU workspace — pre-allocated matrix storage for KA kernel
# =====================================================================

"""
    TM5ConvWorkspace{FT, A4}

Pre-allocated workspace for the TM5 matrix convection GPU kernel.
Holds one lmax x lmax matrix per (i,j) column in `conv1_ws[row, col, i, j]`.
"""
struct TM5ConvWorkspace{FT, A4 <: AbstractArray{FT, 4}}
    conv1_ws::A4
    lmax::Int
end

function allocate_tm5conv_workspace(::Type{FT}, lmax::Int, Ni::Int, Nj::Int,
                                    AT) where FT
    conv1_ws = AT(zeros(FT, lmax, lmax, Ni, Nj))
    return TM5ConvWorkspace(conv1_ws, lmax)
end

const _TM5_CONV_WS_CACHE_LL = Ref{Any}(nothing)
const _TM5_CONV_WS_CACHE_CS = Ref{Any}(nothing)

"""Reset TM5 convection workspace caches."""
function invalidate_tm5_conv_ws!()
    _TM5_CONV_WS_CACHE_LL[] = nothing
    _TM5_CONV_WS_CACHE_CS[] = nothing
    nothing
end

function _get_tm5_conv_ws(cache_ref, ::Type{FT}, lmax::Int, Ni::Int, Nj::Int, AT) where FT
    ws = cache_ref[]
    if ws !== nothing && ws isa TM5ConvWorkspace{FT} && ws.lmax == lmax &&
       size(ws.conv1_ws, 3) == Ni && size(ws.conv1_ws, 4) == Nj
        return ws::TM5ConvWorkspace{FT}
    end
    ws = allocate_tm5conv_workspace(FT, lmax, Ni, Nj, AT)
    cache_ref[] = ws
    mb = round(sizeof(ws.conv1_ws) / 1e6; digits=1)
    @info "Allocated TM5ConvWorkspace: $(lmax)x$(lmax)x$(Ni)x$(Nj) $(FT) ($(mb) MB)" maxlog=3
    return ws
end

# =====================================================================
# KA Kernel: Build TM5 transfer matrix + LU factorize (fused)
#
# Each thread handles one (i,j) column. For each source level kk,
# recomputes fu/fd on the fly (no O(lmax^2) intermediate storage per
# thread beyond the matrix workspace). Then Gaussian elimination
# without pivoting (safe: diagonal-dominant matrix).
# =====================================================================

@kernel function _tm5_conv_build_and_factor_kernel!(
    conv1_ws,
    @Const(entu_data), @Const(detu_data),
    @Const(entd_data), @Const(detd_data),
    @Const(delp),
    i_off, j_off, lmax, Nz, grav, dt
)
    i, j = @index(Global, NTuple)
    ii = i_off + i
    jj = j_off + j
    FT = eltype(conv1_ws)
    ZERO = zero(FT)
    ONE  = one(FT)

    # Initialize matrix to identity
    @inbounds for kk in 1:lmax, k in 1:lmax
        conv1_ws[k, kk, i, j] = (k == kk) ? ONE : ZERO
    end

    # Cloud dimensions (reversed convention)
    li = 0
    @inbounds for k in lmax:-1:1
        if detu_data[ii, jj, Nz + 1 - k] > ZERO; li = k; break; end
    end
    ld = 0
    @inbounds for k in lmax:-1:1
        if entd_data[ii, jj, Nz + 1 - k] > ZERO; ld = k; break; end
    end

    # If no convection, matrix stays as identity — skip build + factorize
    if li > 0 || ld > 0

    # Build conv1 column by column: for each source level kk,
    # recompute fu(interface, kk) bottom-to-top, fd(interface, kk)
    # top-to-bottom, add subsidence, then assemble matrix entries.
    # We use conv1_ws[:, kk, i, j] as temp storage for f_total at
    # interfaces 1..lmax, then overwrite with final matrix entries.

    @inbounds for kk in 1:lmax
        for k in 1:lmax
            conv1_ws[k, kk, i, j] = ZERO
        end

        # --- Updraft: fu at interfaces 1..lmax for source kk ---
        fu_run = ZERO
        amu_run = ZERO
        for k in 1:lmax
            kr = Nz + 1 - k
            entu_k = entu_data[ii, jj, kr]
            detu_k = detu_data[ii, jj, kr]
            m_k = delp[ii, jj, kr] / grav

            if k <= li
                amu_new = amu_run + entu_k - detu_k
                if amu_new > ZERO
                    zxi_u = max(ZERO, ONE - detu_k / (amu_run + entu_k))
                else
                    amu_new = ZERO
                    zxi_u = ZERO
                end
            else
                amu_new = ZERO
                zxi_u = ZERO
            end

            if k < kk || k > li
                fu_k = ZERO
            elseif k == kk
                fu_k = entu_k / m_k * zxi_u
            else
                fu_k = fu_run * zxi_u
            end

            conv1_ws[k, kk, i, j] = fu_k
            fu_run = fu_k
            amu_run = amu_new
        end

        # --- Downdraft: fd at interfaces, added to fu ---
        if kk <= ld && kk >= 2
            fd_run = ZERO
            amd_run = ZERO
            for k in ld:-1:2
                kr = Nz + 1 - k
                entd_k = entd_data[ii, jj, kr]
                detd_k = detd_data[ii, jj, kr]
                m_k = delp[ii, jj, kr] / grav

                amd_new = amd_run - entd_k + detd_k
                if amd_new < ZERO
                    zxi_d = max(ZERO, ONE + detd_k / (amd_run - entd_k))
                else
                    amd_new = ZERO
                    zxi_d = ZERO
                end

                if kk > k
                    fd_new = fd_run * zxi_d
                elseif kk == k
                    fd_new = -entd_k / m_k * zxi_d
                else
                    fd_new = ZERO
                end

                if k - 1 >= 1
                    conv1_ws[k - 1, kk, i, j] += fd_new
                end

                fd_run = fd_new
                amd_run = amd_new
            end
        end

        # --- Subsidence terms ---
        # Updraft subsidence: at interface kk-1, column kk
        if kk >= 2 && kk - 1 <= lmax - 1 && kk - 1 <= li
            amu_sub = ZERO
            for k2 in 1:kk - 1
                k2r = Nz + 1 - k2
                amu_sub = amu_sub + entu_data[ii, jj, k2r] - detu_data[ii, jj, k2r]
                if amu_sub <= ZERO; amu_sub = ZERO; end
            end
            m_kk = delp[ii, jj, Nz + 1 - kk] / grav
            if m_kk > ZERO
                conv1_ws[kk - 1, kk, i, j] -= amu_sub / m_kk
            end
        end

        # Downdraft subsidence: at interface kk, column kk
        if kk >= 1 && kk <= lmax - 1 && kk < ld
            amd_sub = ZERO
            for k2 in ld:-1:kk + 1
                k2r = Nz + 1 - k2
                amd_sub = amd_sub - entd_data[ii, jj, k2r] + detd_data[ii, jj, k2r]
                if amd_sub >= ZERO; amd_sub = ZERO; end
            end
            m_kk = delp[ii, jj, Nz + 1 - kk] / grav
            if m_kk > ZERO && amd_sub != ZERO
                conv1_ws[kk, kk, i, j] -= amd_sub / m_kk
            end
        end

        # --- Assemble conv1[k, kk] from f_total differences ---
        for k in lmax:-1:1
            f_at_k = conv1_ws[k, kk, i, j]
            f_below_k = (k > 1) ? conv1_ws[k - 1, kk, i, j] : ZERO
            entry = -dt * (f_below_k - f_at_k)
            if k == kk; entry += ONE; end
            conv1_ws[k, kk, i, j] = entry
        end
    end

    # --- Gaussian elimination without pivoting ---
    @inbounds for p in 1:lmax - 1
        pivot = conv1_ws[p, p, i, j]
        abs(pivot) < FT(1e-30) && continue
        inv_pivot = ONE / pivot
        for k in p + 1:lmax
            factor = conv1_ws[k, p, i, j] * inv_pivot
            factor == ZERO && continue
            conv1_ws[k, p, i, j] = factor
            for kk in p + 1:lmax
                conv1_ws[k, kk, i, j] -= factor * conv1_ws[p, kk, i, j]
            end
        end
    end

    end # if li > 0 || ld > 0
end

# =====================================================================
# KA Kernel: Forward/back substitution using pre-computed LU
# =====================================================================

@kernel function _tm5_conv_solve_kernel!(
    arr, @Const(m_arr),
    @Const(conv1_ws),
    @Const(delp),
    i_off, j_off, lmax, Nz, grav,
    ::Val{tracer_mode}
) where tracer_mode
    i, j = @index(Global, NTuple)
    ii = i_off + i
    jj = j_off + j
    FT = eltype(arr)
    ZERO = zero(FT)

    # Forward substitution: L * y = b
    # b[k] = arr * m (mixing ratio -> rm) in TM5 convention
    @inbounds for k in 1:lmax
        kr = Nz + 1 - k
        if tracer_mode === :mixing_ratio
            m_k = delp[ii, jj, kr] / grav
            val = arr[ii, jj, kr] * m_k
        else
            val = arr[ii, jj, kr]
        end
        for p in 1:k - 1
            pr = Nz + 1 - p
            L_kp = conv1_ws[k, p, i, j]
            if L_kp != ZERO
                val -= L_kp * arr[ii, jj, pr]
            end
        end
        arr[ii, jj, kr] = val
    end

    # Back substitution: U * x = y
    @inbounds for k in lmax:-1:1
        kr = Nz + 1 - k
        val = arr[ii, jj, kr]
        for kk in k + 1:lmax
            kkr = Nz + 1 - kk
            U_kkk = conv1_ws[k, kk, i, j]
            if U_kkk != ZERO
                val -= U_kkk * arr[ii, jj, kkr]
            end
        end
        arr[ii, jj, kr] = val / conv1_ws[k, k, i, j]
    end

    # rm -> mixing ratio
    if tracer_mode === :mixing_ratio
        @inbounds for k in 1:lmax
            kr = Nz + 1 - k
            m_k = delp[ii, jj, kr] / grav
            arr[ii, jj, kr] = m_k > ZERO ? arr[ii, jj, kr] / m_k : ZERO
        end
    end
end

# =====================================================================
# GPU dispatch helper
# =====================================================================

function _tm5_conv_gpu!(tracers, entu, detu, entd, detd, delp,
                        ws::TM5ConvWorkspace, backend,
                        lmax, Ni, Nj, Nz, grav::FT, dt::FT;
                        i_off=0, j_off=0,
                        tracer_mode=:mixing_ratio) where FT
    build_k! = _tm5_conv_build_and_factor_kernel!(backend, 256)
    build_k!(ws.conv1_ws, entu, detu, entd, detd, delp,
             i_off, j_off, lmax, Nz, grav, dt; ndrange=(Ni, Nj))
    synchronize(backend)

    solve_k! = _tm5_conv_solve_kernel!(backend, 256)
    m_dummy = delp
    tracer_list = tracers isa NamedTuple ? values(tracers) : tracers
    for tracer in tracer_list
        arr = _tm5_tracer_data(tracer)
        solve_k!(arr, m_dummy, ws.conv1_ws, delp,
                 i_off, j_off, lmax, Nz, grav,
                 Val(tracer_mode); ndrange=(Ni, Nj))
        synchronize(backend)
    end
end

# =====================================================================
# Dispatch: LatitudeLongitudeGrid
# =====================================================================

"""
    convect!(tracers, tm5conv_data, delp, conv::TM5MatrixConvection,
             grid::LatitudeLongitudeGrid, dt, planet; kwargs...)

Apply TM5 matrix convection to lat-lon tracers.
GPU-aware: uses KA kernels on GPU, threaded LAPACK LU on CPU.
"""
function convect!(tracers::NamedTuple, tm5conv_data::NamedTuple, delp,
                   conv::TM5MatrixConvection, grid::LatitudeLongitudeGrid, dt,
                   planet::PlanetParameters;
                   dtrain_panels=nothing, workspace=nothing)
    FT = floattype(grid)
    Nx, Ny, Nz = size(delp)
    grav = FT(planet.gravity)
    dt_FT = FT(dt)
    lmax = conv.lmax_conv > 0 ? min(conv.lmax_conv, Nz) : Nz

    entu_data = tm5conv_data.entu
    detu_data = tm5conv_data.detu
    entd_data = tm5conv_data.entd
    detd_data = tm5conv_data.detd

    if delp isa Array
        _tm5_conv_cpu!(tracers, entu_data, detu_data, entd_data, detd_data,
                       delp, lmax, Nx, Ny, Nz, grav, dt_FT)
    else
        backend = get_backend(delp)
        AT = typeof(delp).name.wrapper
        ws = if workspace !== nothing && workspace isa TM5ConvWorkspace
            workspace
        else
            _get_tm5_conv_ws(_TM5_CONV_WS_CACHE_LL, FT, lmax, Nx, Ny, AT)
        end
        _tm5_conv_gpu!(tracers, entu_data, detu_data, entd_data, detd_data,
                       delp, ws, backend, lmax, Nx, Ny, Nz, grav, dt_FT;
                       i_off=0, j_off=0, tracer_mode=:mixing_ratio)
    end
    return nothing
end

# =====================================================================
# CPU path — threaded per-column LU solve
# =====================================================================

function _tm5_conv_cpu!(tracers, entu_data, detu_data, entd_data, detd_data,
                        delp, lmax, Nx, Ny, Nz, grav::FT, dt_FT::FT) where FT
    Threads.@threads for idx in 1:Nx*Ny
        j = div(idx - 1, Nx) + 1
        i = mod(idx - 1, Nx) + 1

        m_col     = Vector{FT}(undef, lmax)
        entu_col  = Vector{FT}(undef, lmax)
        detu_col  = Vector{FT}(undef, lmax)
        entd_col  = Vector{FT}(undef, lmax)
        detd_col  = Vector{FT}(undef, lmax)
        conv1     = Matrix{FT}(undef, lmax, lmax)

        @inbounds for k in 1:lmax
            k_rev = Nz + 1 - k
            m_col[k]    = delp[i, j, k_rev] / grav
            entu_col[k] = entu_data[i, j, k_rev]
            detu_col[k] = detu_data[i, j, k_rev]
            entd_col[k] = entd_data[i, j, k_rev]
            detd_col[k] = detd_data[i, j, k_rev]
        end

        li, ld = _conv_cloud_dim(detu_col, entd_col, lmax)
        (li == 0 && ld == 0) && continue

        lmc = tm5_conv_matrix!(conv1, m_col, entu_col, detu_col,
                               entd_col, detd_col, lmax, li, ld, dt_FT)
        lmc == 0 && continue

        conv1_sub = @view conv1[1:lmc, 1:lmc]
        F = lu!(conv1_sub)

        for tracer in values(tracers)
            arr = _tm5_tracer_data(tracer)
            rm_col = Vector{FT}(undef, lmc)
            @inbounds for k in 1:lmc
                k_rev = Nz + 1 - k
                rm_col[k] = arr[i, j, k_rev] * m_col[k]
            end
            ldiv!(F, rm_col)
            @inbounds for k in 1:lmc
                k_rev = Nz + 1 - k
                arr[i, j, k_rev] = rm_col[k] / m_col[k]
            end
        end
    end
end

# =====================================================================
# Dispatch: CubedSphereGrid — loop over 6 haloed panels
# =====================================================================

"""
    convect!(rm_panels, m_panels, tm5conv_panels, delp_panels,
             conv::TM5MatrixConvection, grid::CubedSphereGrid, dt, planet; kwargs...)

Apply TM5 matrix convection to cubed-sphere panel arrays.
Each panel's tracer mass (rm) is converted to mixing ratio, convected
via TM5's implicit matrix scheme, then converted back.
"""
function convect!(rm_panels::NTuple{6}, m_panels::NTuple{6},
                   tm5conv_panels::NamedTuple, delp_panels::NTuple{6},
                   conv::TM5MatrixConvection, grid::CubedSphereGrid, dt,
                   planet::PlanetParameters;
                   dtrain_panels=nothing, workspace=nothing)
    FT = eltype(rm_panels[1])
    Nc = grid.Nc
    Hp = grid.Hp
    Nz = grid.Nz
    grav = FT(planet.gravity)
    dt_FT = FT(dt)
    lmax = conv.lmax_conv > 0 ? min(conv.lmax_conv, Nz) : Nz

    AT = typeof(rm_panels[1]).name.wrapper
    ws = _get_tm5_conv_ws(_TM5_CONV_WS_CACHE_CS, FT, lmax, Nc, Nc, AT)

    for_panels_nosync() do p
        backend = get_backend(rm_panels[p])

        # rm -> q (mixing ratio) in-place
        _rm_to_q_k! = _rm_to_q_column_kernel!(backend, 256)
        _rm_to_q_k!(rm_panels[p], m_panels[p], Hp, Nz; ndrange=(Nc * Nc,))

        # Build + factorize
        build_k! = _tm5_conv_build_and_factor_kernel!(backend, 256)
        build_k!(ws.conv1_ws,
                 tm5conv_panels.entu[p], tm5conv_panels.detu[p],
                 tm5conv_panels.entd[p], tm5conv_panels.detd[p],
                 delp_panels[p],
                 Hp, Hp, lmax, Nz, grav, dt_FT; ndrange=(Nc, Nc))

        # Solve (arr in mixing-ratio space)
        solve_k! = _tm5_conv_solve_kernel!(backend, 256)
        solve_k!(rm_panels[p], m_panels[p], ws.conv1_ws, delp_panels[p],
                 Hp, Hp, lmax, Nz, grav,
                 Val(:mixing_ratio); ndrange=(Nc, Nc))

        # q -> rm in-place
        _q_to_rm_k! = _q_to_rm_column_kernel!(backend, 256)
        _q_to_rm_k!(rm_panels[p], m_panels[p], Hp, Nz; ndrange=(Nc * Nc,))
    end
    return nothing
end

# Helper KA kernels for rm <-> q conversion on CS panels

@kernel function _rm_to_q_column_kernel!(arr, @Const(m), Hp, Nz)
    idx = @index(Global, Linear)
    FT = eltype(arr)
    Nc_dim = size(arr, 1) - 2 * Hp
    i_local = mod(idx - 1, Nc_dim) + 1 + Hp
    j_local = div(idx - 1, Nc_dim) + 1 + Hp
    @inbounds for k in 1:Nz
        _m = m[i_local, j_local, k]
        arr[i_local, j_local, k] = _m > FT(1e-30) ? arr[i_local, j_local, k] / _m : zero(FT)
    end
end

@kernel function _q_to_rm_column_kernel!(arr, @Const(m), Hp, Nz)
    idx = @index(Global, Linear)
    Nc_dim = size(arr, 1) - 2 * Hp
    i_local = mod(idx - 1, Nc_dim) + 1 + Hp
    j_local = div(idx - 1, Nc_dim) + 1 + Hp
    @inbounds for k in 1:Nz
        arr[i_local, j_local, k] *= m[i_local, j_local, k]
    end
end
