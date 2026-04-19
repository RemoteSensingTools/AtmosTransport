"""
    _vertical_diffusion_kernel!(q, kz_field, dz, w_scratch, dt, Nz)

KernelAbstractions kernel: implicit (Backward-Euler) vertical
diffusion for one column per `(i, j, t)` thread.

- `q::AbstractArray{FT, 4}` — tracer values `(Nx, Ny, Nz, Nt)`,
  read for old values and written with new values in place.
- `kz_field::AbstractTimeVaryingField{FT, 3}` — Kz at cell centers.
- `dz::AbstractArray{FT, 3}` — layer thicknesses in meters,
  `(Nx, Ny, Nz)`. Caller supplies; not mutated.
- `w_scratch::AbstractArray{FT, 3}` — caller-supplied workspace,
  `(Nx, Ny, Nz)`. Holds the Thomas forward-elimination factors
  between the forward and back-substitution loops.
- `dt::FT` — time step.
- `Nz::Int` — number of vertical levels (passed explicitly so the
  kernel avoids a `size` call).

The `(a, b, c, d)` tridiagonal entries are **named local values at
each level k** rather than pre-built arrays. The coefficient
formulas exactly match [`build_diffusion_coefficients`](@ref) —
the reference used by the test suite. Any change to the formulas
here requires updating the reference (or vice versa).

# Adjoint note

The future adjoint kernel is structurally identical:
read Kz and dz the same way, build the same `(a_k, b_k, c_k)`
per level, then apply the transposition rule at the tridiagonal
interface (`a_T[k] = c[k-1]`, `b_T[k] = b[k]`, `c_T[k] = a[k+1]`)
before passing to a Thomas solve. See
[src_legacy/Diffusion/boundary_layer_diffusion_adjoint.jl:74-84](src_legacy/Diffusion/boundary_layer_diffusion_adjoint.jl#L74-L84).
"""
@kernel function _vertical_diffusion_kernel!(q, kz_field,
                                              @Const(dz),
                                              w_scratch,
                                              dt, Nz::Int)
    i, j, t = @index(Global, NTuple)
    FT = eltype(q)
    @inbounds begin
        dt_ft = FT(dt)

        # --- Loop-persistent caches.  Pre-declared so the analyzer
        # (and a human reader) can see the data flow across iterations. ---
        Kz_prev = zero(FT)
        dz_prev = zero(FT)
        w_prev  = zero(FT)
        g_prev  = zero(FT)

        # Read level-1 Kz and dz outside the loop so each iteration's
        # `Kz_next` / `dz_next` reads can feed forward without re-reading
        # the same slot twice.
        Kz_k = field_value(kz_field, (i, j, 1))
        dz_k = dz[i, j, 1]

        for k in 1:Nz
            # --- Coefficient construction at level k. ---
            # Interface Kz and dz: arithmetic mean of adjacent centers.
            # k = 1   → no cell above  → D_above = 0
            # k = Nz  → no cell below  → D_below = 0
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

            # Tridiagonal entry at level k (named locals, not fused).
            a_k = (k > 1)  ? -dt_ft * D_above : zero(FT)
            b_k = one(FT) + dt_ft * (D_above + D_below)
            c_k = (k < Nz) ? -dt_ft * D_below : zero(FT)
            d_k = q[i, j, k, t]

            # --- Thomas forward elimination. ---
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
            q[i, j, k, t]      = g_k      # temporarily store g_k here

            # --- Shuffle caches for the next iteration. Skip on k = Nz
            # so the static analyzer doesn't flag dead stores. ---
            if k < Nz
                w_prev  = w_k
                g_prev  = g_k
                Kz_prev = Kz_k
                dz_prev = dz_k
                Kz_k    = Kz_next
                dz_k    = dz_next
            end
        end

        # --- Back-substitution: x[k] = g_k - w[k] * x[k+1] ---
        # q[:, :, Nz, t] already holds x[Nz] = g[Nz].
        for k in (Nz - 1):-1:1
            q[i, j, k, t] = q[i, j, k, t] - w_scratch[i, j, k] * q[i, j, k + 1, t]
        end
    end
end
