# ---------------------------------------------------------------------------
# Vertical diffusion kernels — mass-flux form, TM5-style.
#
# Backward-Euler implicit solve `Ã·q_new = q_old` on dry mass-mixing
# ratio `q`. The coefficient matrix `Ã` is built so that the equivalent
# operator on tracer mass `rm = m·q` is column-stochastic (each column
# sums to 1), which preserves `Σ m·q` to roundoff for any inert tracer
# — the conservation bar that the convection sweep also aims at.
#
# Coefficients per level k (with `m_k = air_mass[..., k]`, `Kz` at cell
# centers, `dz` thickness in meters):
#
#     dkg[k-½] = (m_{k-1} + m_k) · (Kz_{k-1} + Kz_k) / (dz_{k-1} + dz_k)²
#     dkg[k+½] = (m_k + m_{k+1}) · (Kz_k + Kz_{k+1}) / (dz_k + dz_{k+1})²
#
#     a_k = -dt · dkg[k-½] / m_k                       (sub-diagonal)
#     b_k = 1 + dt · (dkg[k-½] + dkg[k+½]) / m_k       (diagonal)
#     c_k = -dt · dkg[k+½] / m_k                       (super-diagonal)
#
# Boundary: dkg[½] = dkg[Nz+½] = 0 (zero-flux at TOA and surface).
#
# Reference: TM5's `TM5_Diff_Matrix` at `deps/tm5-cy3-4dvar/base/src/
# tm5_diff.F90:36-129` uses the same `fu = dt·dkg/m(k)`, `fd = dt·dkg/m(k+1)`
# fraction-of-mass-exchanged form on the tracer-mass vector directly.
# We operate on VMR with `Ã = M⁻¹·A·M`, which has the same conservation
# guarantee via `Σ m·q_new = Σ m·q_old`. See the algebra in
# `memory/diffusion_full_pipeline_audit_2026_05_25.md` (D1 fix).
#
# The previous geometric form `D = Kz / (dz_k · dz_iface)` (kept in git
# history at commit 400410b~) conserved `Σ q·dz` but not `Σ m·q`,
# leaking tracer mass by ~1–10 % per 3-day CS180 run.
# ---------------------------------------------------------------------------

"""
    _vertical_diffusion_kernel!(q, air_mass, kz_field, dz, w_scratch, dt, Nz)

KernelAbstractions kernel: implicit (Backward-Euler) vertical
diffusion for one column per `(i, j, t)` thread, mass-flux form
(TM5-style; preserves `Σ m·q` to roundoff).

- `q::AbstractArray{FT, 4}` — dry-VMR tracer values `(Nx, Ny, Nz, Nt)`,
  read for old values and written with new values in place.
- `air_mass::AbstractArray{FT, 3}` — dry layer mass `(Nx, Ny, Nz)`,
  used to build the mass-flux coefficients. Not mutated.
- `kz_field::AbstractTimeVaryingField{FT, 3}` — Kz at cell centers.
- `dz::AbstractArray{FT, 3}` — layer thicknesses in meters,
  `(Nx, Ny, Nz)`. Caller supplies; not mutated.
- `w_scratch::AbstractArray{FT, 3}` — caller-supplied workspace,
  `(Nx, Ny, Nz)`. Holds the Thomas forward-elimination factors
  between the forward and back-substitution loops.
- `dt::FT` — time step.
- `Nz::Int` — number of vertical levels.

# Adjoint note

The mass-flux coefficients are not symmetric in `(a, c)`: at row k,
`a_k = -dt·dkg[k-½]/m_k` and `c_k = -dt·dkg[k+½]/m_k`, with `m_k` in
both denominators (NOT `m_{k-1}` and `m_{k+1}`). The TRANSPOSE of this
matrix swaps to `a_T[k] = c[k-1]·... ` patterns that resolve to
`-dt·dkg[k-½]/m_{k-1}` etc. See `src/Adjoints/DiffusionAdjoint.jl` for
the corresponding adjoint kernel that reads `air_mass` and builds the
transposed coefficient form.
"""
# NOTE: LL packed and RG (face-indexed) kernels remain on the legacy
# GEOMETRIC form `D = Kz/(dz·dz)` for now. The mass-flux conservation
# rewrite has only been threaded through the CS path because that's
# where the user's experiments run; the LL/RG paths currently apply
# the kernel directly to tracer mass without pre/post VMR scaling and
# fixing them requires either (a) adding the LL/RG mass-VMR wrapper
# (matching CS's `apply_vertical_diffusion_vmr!`) or (b) reformulating
# the LL/RG state contract to VMR-native. Tracked as a follow-up in
# the D1 audit memo. See the CS-side kernels below for the mass-flux
# form.
@kernel function _vertical_diffusion_kernel!(q, kz_field,
                                              @Const(dz),
                                              w_scratch,
                                              dt, Nz::Int)
    i, j, t = @index(Global, NTuple)
    FT = eltype(q)
    @inbounds begin
        dt_ft = FT(dt)

        Kz_prev = zero(FT)
        dz_prev = zero(FT)
        w_prev  = zero(FT)
        g_prev  = zero(FT)

        Kz_k = field_value(kz_field, (i, j, 1))
        dz_k = dz[i, j, 1]

        for k in 1:Nz
            D_above = zero(FT)
            D_below = zero(FT)
            Kz_next = zero(FT)
            dz_next = zero(FT)

            if k > 1
                Kz_above = (Kz_prev + Kz_k) / FT(2)
                dz_above = (dz_prev + dz_k) / FT(2)
                D_above  = Kz_above / (dz_k * dz_above)
            end

            if k < Nz
                Kz_next  = field_value(kz_field, (i, j, k + 1))
                dz_next  = dz[i, j, k + 1]
                Kz_below = (Kz_k + Kz_next) / FT(2)
                dz_below = (dz_k + dz_next) / FT(2)
                D_below  = Kz_below / (dz_k * dz_below)
            end

            a_k = (k > 1)  ? -dt_ft * D_above : zero(FT)
            b_k = one(FT) + dt_ft * (D_above + D_below)
            c_k = (k < Nz) ? -dt_ft * D_below : zero(FT)
            d_k = q[i, j, k, t]

            if k == 1
                denom = b_k
                w_k   = c_k / denom
                g_k   = d_k / denom
            else
                denom = b_k - a_k * w_prev
                w_k   = c_k / denom
                g_k   = (d_k - a_k * g_prev) / denom
            end

            w_scratch[i, j, k] = w_k
            q[i, j, k, t]      = g_k

            if k < Nz
                w_prev  = w_k
                g_prev  = g_k
                Kz_prev = Kz_k
                dz_prev = dz_k
                Kz_k    = Kz_next
                dz_k    = dz_next
            end
        end

        for k in (Nz - 1):-1:1
            q[i, j, k, t] = q[i, j, k, t] - w_scratch[i, j, k] * q[i, j, k + 1, t]
        end
    end
end

"""
    _vertical_diffusion_cs_single_kernel!(q, kz_field, dz, w_scratch, dt, Nz, Hp)

Cubed-sphere single-tracer diffusion kernel.

`q` is one halo-padded panel `(Nc + 2Hp, Nc + 2Hp, Nz)` while `dz` and
`w_scratch` are interior `(Nc, Nc, Nz)` workspaces. The structured
column solve is unchanged; only the panel halo offset differs.
"""
@kernel function _vertical_diffusion_cs_single_kernel!(q, @Const(air_mass),
                                                       kz_field,
                                                       @Const(dz),
                                                       w_scratch,
                                                       dt, Nz::Int, Hp::Int)
    ii, jj = @index(Global, NTuple)
    FT = eltype(q)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        dt_ft = FT(dt)

        Kz_prev = zero(FT)
        dz_prev = zero(FT)
        m_prev  = zero(FT)
        w_prev  = zero(FT)
        g_prev  = zero(FT)

        Kz_k = field_value(kz_field, (ii, jj, 1))
        dz_k = dz[ii, jj, 1]
        m_k  = air_mass[i, j, 1]

        for k in 1:Nz
            dkg_above = zero(FT)
            dkg_below = zero(FT)
            Kz_next   = zero(FT)
            dz_next   = zero(FT)
            m_next    = zero(FT)

            if k > 1
                sum_dz_above = dz_prev + dz_k
                dkg_above = (m_prev + m_k) * (Kz_prev + Kz_k) /
                            (sum_dz_above * sum_dz_above)
            end

            if k < Nz
                Kz_next  = field_value(kz_field, (ii, jj, k + 1))
                dz_next  = dz[ii, jj, k + 1]
                m_next   = air_mass[i, j, k + 1]
                sum_dz_below = dz_k + dz_next
                dkg_below = (m_k + m_next) * (Kz_k + Kz_next) /
                            (sum_dz_below * sum_dz_below)
            end

            inv_m_k = m_k > zero(FT) ? one(FT) / m_k : zero(FT)
            a_k = (k > 1)  ? -dt_ft * dkg_above * inv_m_k : zero(FT)
            c_k = (k < Nz) ? -dt_ft * dkg_below * inv_m_k : zero(FT)
            b_k = one(FT) + dt_ft * (dkg_above + dkg_below) * inv_m_k
            d_k = q[i, j, k]

            if k == 1
                denom = b_k
                w_k   = c_k / denom
                g_k   = d_k / denom
            else
                denom = b_k - a_k * w_prev
                w_k   = c_k / denom
                g_k   = (d_k - a_k * g_prev) / denom
            end

            w_scratch[ii, jj, k] = w_k
            q[i, j, k]           = g_k

            if k < Nz
                w_prev  = w_k
                g_prev  = g_k
                Kz_prev = Kz_k
                dz_prev = dz_k
                m_prev  = m_k
                Kz_k    = Kz_next
                dz_k    = dz_next
                m_k     = m_next
            end
        end

        for k in (Nz - 1):-1:1
            q[i, j, k] = q[i, j, k] - w_scratch[ii, jj, k] * q[i, j, k + 1]
        end
    end
end

"""
    _vertical_diffusion_cs_kernel!(q, kz_field, dz, w_scratch, dt, Nz, Hp)

Packed cubed-sphere diffusion kernel. `q` is one halo-padded panel
`(Nc + 2Hp, Nc + 2Hp, Nz, Nt)`. One thread owns one `(ii, jj, tracer)`
column; the tridiagonal coefficients are identical for all tracers in a
column and are computed locally to keep the workspace rank unchanged.
"""
@kernel function _vertical_diffusion_cs_kernel!(q, @Const(air_mass),
                                                kz_field,
                                                @Const(dz),
                                                w_scratch,
                                                dt, Nz::Int, Hp::Int)
    ii, jj, t = @index(Global, NTuple)
    FT = eltype(q)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        dt_ft = FT(dt)

        Kz_prev = zero(FT)
        dz_prev = zero(FT)
        m_prev  = zero(FT)
        w_prev  = zero(FT)
        g_prev  = zero(FT)

        Kz_k = field_value(kz_field, (ii, jj, 1))
        dz_k = dz[ii, jj, 1]
        m_k  = air_mass[i, j, 1]

        for k in 1:Nz
            dkg_above = zero(FT)
            dkg_below = zero(FT)
            Kz_next   = zero(FT)
            dz_next   = zero(FT)
            m_next    = zero(FT)

            if k > 1
                sum_dz_above = dz_prev + dz_k
                dkg_above = (m_prev + m_k) * (Kz_prev + Kz_k) /
                            (sum_dz_above * sum_dz_above)
            end

            if k < Nz
                Kz_next  = field_value(kz_field, (ii, jj, k + 1))
                dz_next  = dz[ii, jj, k + 1]
                m_next   = air_mass[i, j, k + 1]
                sum_dz_below = dz_k + dz_next
                dkg_below = (m_k + m_next) * (Kz_k + Kz_next) /
                            (sum_dz_below * sum_dz_below)
            end

            inv_m_k = m_k > zero(FT) ? one(FT) / m_k : zero(FT)
            a_k = (k > 1)  ? -dt_ft * dkg_above * inv_m_k : zero(FT)
            c_k = (k < Nz) ? -dt_ft * dkg_below * inv_m_k : zero(FT)
            b_k = one(FT) + dt_ft * (dkg_above + dkg_below) * inv_m_k
            d_k = q[i, j, k, t]

            if k == 1
                denom = b_k
                w_k   = c_k / denom
                g_k   = d_k / denom
            else
                denom = b_k - a_k * w_prev
                w_k   = c_k / denom
                g_k   = (d_k - a_k * g_prev) / denom
            end

            w_scratch[ii, jj, k] = w_k
            q[i, j, k, t]        = g_k

            if k < Nz
                w_prev  = w_k
                g_prev  = g_k
                Kz_prev = Kz_k
                dz_prev = dz_k
                m_prev  = m_k
                Kz_k    = Kz_next
                dz_k    = dz_next
                m_k     = m_next
            end
        end

        for k in (Nz - 1):-1:1
            q[i, j, k, t] = q[i, j, k, t] - w_scratch[ii, jj, k] * q[i, j, k + 1, t]
        end
    end
end

@kernel function _cs_tracer_mass_to_vmr_kernel!(q, @Const(air_mass), Hp::Int)
    ii, jj, k = @index(Global, NTuple)
    FT = eltype(q)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        m = air_mass[i, j, k]
        q[i, j, k] = m > zero(FT) ? q[i, j, k] / m : zero(FT)
    end
end

@kernel function _cs_vmr_to_tracer_mass_kernel!(q, @Const(air_mass), Hp::Int)
    ii, jj, k = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        q[i, j, k] *= air_mass[i, j, k]
    end
end

@kernel function _cs_tracer_mass_to_vmr_4d_kernel!(q, @Const(air_mass), Hp::Int)
    ii, jj, k, t = @index(Global, NTuple)
    FT = eltype(q)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        m = air_mass[i, j, k]
        q[i, j, k, t] = m > zero(FT) ? q[i, j, k, t] / m : zero(FT)
    end
end

@kernel function _cs_vmr_to_tracer_mass_4d_kernel!(q, @Const(air_mass), Hp::Int)
    ii, jj, k, t = @index(Global, NTuple)
    @inbounds begin
        i = ii + Hp
        j = jj + Hp
        q[i, j, k, t] *= air_mass[i, j, k]
    end
end

"""
    _vertical_diffusion_face_kernel!(q, kz_field, dz, w_scratch, dt, Nz)

Face-indexed vertical diffusion kernel. One thread per `(cell, tracer)`
column on a packed `(ncells, Nz, Nt)` tracer array. The arithmetic is
identical to `_vertical_diffusion_kernel!`; only the storage
layout changes.
"""
@kernel function _vertical_diffusion_face_kernel!(q, kz_field,
                                                  @Const(dz),
                                                  w_scratch,
                                                  dt, Nz::Int)
    c, t = @index(Global, NTuple)
    FT = eltype(q)
    @inbounds begin
        dt_ft = FT(dt)

        Kz_prev = zero(FT)
        dz_prev = zero(FT)
        w_prev  = zero(FT)
        g_prev  = zero(FT)

        Kz_k = field_value(kz_field, (c, 1))
        dz_k = dz[c, 1]

        for k in 1:Nz
            D_above = zero(FT)
            D_below = zero(FT)
            Kz_next = zero(FT)
            dz_next = zero(FT)

            if k > 1
                Kz_above = (Kz_prev + Kz_k) / FT(2)
                dz_above = (dz_prev + dz_k) / FT(2)
                D_above  = Kz_above / (dz_k * dz_above)
            end

            if k < Nz
                Kz_next  = field_value(kz_field, (c, k + 1))
                dz_next  = dz[c, k + 1]
                Kz_below = (Kz_k + Kz_next) / FT(2)
                dz_below = (dz_k + dz_next) / FT(2)
                D_below  = Kz_below / (dz_k * dz_below)
            end

            a_k = (k > 1)  ? -dt_ft * D_above : zero(FT)
            b_k = one(FT) + dt_ft * (D_above + D_below)
            c_k = (k < Nz) ? -dt_ft * D_below : zero(FT)
            d_k = q[c, k, t]

            if k == 1
                denom = b_k
                w_k   = c_k / denom
                g_k   = d_k / denom
            else
                denom = b_k - a_k * w_prev
                w_k   = c_k / denom
                g_k   = (d_k - a_k * g_prev) / denom
            end

            w_scratch[c, k] = w_k
            q[c, k, t]      = g_k

            if k < Nz
                w_prev  = w_k
                g_prev  = g_k
                Kz_prev = Kz_k
                dz_prev = dz_k
                Kz_k    = Kz_next
                dz_k    = dz_next
            end
        end

        for k in (Nz - 1):-1:1
            q[c, k, t] = q[c, k, t] - w_scratch[c, k] * q[c, k + 1, t]
        end
    end
end

"""
    _vertical_diffusion_face_single_kernel!(q, kz_field, dz, w_scratch, dt, Nz)

Face-indexed single-tracer helper operating on a `(ncells, Nz)` tracer
slice. Used by the reduced-Gaussian advection palindrome, which keeps a
per-tracer host loop.
"""
@kernel function _vertical_diffusion_face_single_kernel!(q, kz_field,
                                                         @Const(dz),
                                                         w_scratch,
                                                         dt, Nz::Int)
    c = @index(Global, Linear)
    FT = eltype(q)
    @inbounds begin
        dt_ft = FT(dt)

        Kz_prev = zero(FT)
        dz_prev = zero(FT)
        w_prev  = zero(FT)
        g_prev  = zero(FT)

        Kz_k = field_value(kz_field, (c, 1))
        dz_k = dz[c, 1]

        for k in 1:Nz
            D_above = zero(FT)
            D_below = zero(FT)
            Kz_next = zero(FT)
            dz_next = zero(FT)

            if k > 1
                Kz_above = (Kz_prev + Kz_k) / FT(2)
                dz_above = (dz_prev + dz_k) / FT(2)
                D_above  = Kz_above / (dz_k * dz_above)
            end

            if k < Nz
                Kz_next  = field_value(kz_field, (c, k + 1))
                dz_next  = dz[c, k + 1]
                Kz_below = (Kz_k + Kz_next) / FT(2)
                dz_below = (dz_k + dz_next) / FT(2)
                D_below  = Kz_below / (dz_k * dz_below)
            end

            a_k = (k > 1)  ? -dt_ft * D_above : zero(FT)
            b_k = one(FT) + dt_ft * (D_above + D_below)
            c_k = (k < Nz) ? -dt_ft * D_below : zero(FT)
            d_k = q[c, k]

            if k == 1
                denom = b_k
                w_k   = c_k / denom
                g_k   = d_k / denom
            else
                denom = b_k - a_k * w_prev
                w_k   = c_k / denom
                g_k   = (d_k - a_k * g_prev) / denom
            end

            w_scratch[c, k] = w_k
            q[c, k]         = g_k

            if k < Nz
                w_prev  = w_k
                g_prev  = g_k
                Kz_prev = Kz_k
                dz_prev = dz_k
                Kz_k    = Kz_next
                dz_k    = dz_next
            end
        end

        for k in (Nz - 1):-1:1
            q[c, k] = q[c, k] - w_scratch[c, k] * q[c, k + 1]
        end
    end
end
