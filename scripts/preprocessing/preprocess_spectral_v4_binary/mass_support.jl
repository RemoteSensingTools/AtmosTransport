"""
    merge_cell_field!(merged, native, mm)

Accumulate a native-level cell field onto merged output levels using the
native-to-merged level map `mm`.
"""
function merge_cell_field!(merged::Array{FT, 3}, native::Array{FT, 3}, mm::Vector{Int}) where FT
    Threads.@threads for km in 1:size(merged, 3)
        @views merged[:, :, km] .= zero(FT)
    end
    @inbounds for k in 1:length(mm)
        @views merged[:, :, mm[k]] .+= native[:, :, k]
    end
end

"""
    balance_mass_fluxes!(am, bm, dm_dt)

TM5-style Poisson mass-flux balance (TM5 r1112 grid_type_ll.F90:2536-2653).

Adjusts `am` and `bm` in-place so that the discrete horizontal convergence
at each cell `(i, j)` matches the prescribed mass tendency `dm_dt`:

    (am[i] − am[i+1]) + (bm[j] − bm[j+1])  =  dm_dt[i, j]     ∀ (i, j)

The correction is a streamfunction `ψ` found by solving the 2D discrete
Poisson equation `∇²ψ = residual` on the periodic lat-lon grid via FFT.

## Algorithm

For each level `k`:

1. Compute the residual: `r[i, j] = convergence[i, j] − dm_dt[i, j]`.
2. Forward FFT: `R̂ = FFT(r)`.
3. Divide by the discrete Laplacian eigenvalues:
   `fac[i, j] = 2 (cos(2π(i−1)/Nx) + cos(2π(j−1)/Ny) − 2)`
   which are the eigenvalues of the 2D circulant Laplacian with periodic
   BCs in both lon and lat. The (1,1) mode (global mean) is set to 1
   to avoid division by zero (constant null space).
4. Inverse FFT: `ψ = real(IFFT(R̂ / fac))`.
5. Apply flux correction from the gradient of ψ:
   `am[i, j] += ψ[i] − ψ[i−1]` (zonal)
   `bm[i, j] += ψ[j] − ψ[j−1]` (meridional)
   with periodic wrapping in both directions.

After this, the horizontal convergence at each cell equals `dm_dt` to
machine precision (~1e-2 kg residual at F64). The residual is the
GLOBAL mean of `r`, which cannot be corrected by divergence-free
adjustments — it's absorbed by the downstream `cm` cumsum. See
CLAUDE.md invariant #13 for the full discussion.

## The dm_dt target

The target is computed by `fill_window_mass_tendency!` as:

    dm_dt[i, j, k] = (m_next − m_cur)[i, j, k] / (2 × steps_per_window)

The factor of 2 is because the Strang splitting applies horizontal
fluxes twice per full step (forward X-Y-Z + reverse Z-Y-X), so the
per-application mass tendency is half the total window tendency. This
is TM5's `poisson_balance_target_scale`.

## Latitude BC note

The FFT solver assumes periodicity in BOTH lon and lat. The lat wrap
(south pole ↔ north pole) is unphysical but harmless because the
pole-row fluxes are forced to zero immediately after the balance pass
(see `apply_structured_pole_constraints!` in binary_pipeline.jl).
"""
function balance_mass_fluxes!(am::Array{FT, 3}, bm::Array{FT, 3},
                              dm_dt::Array{FT, 3}) where FT
    Nx = size(am, 1) - 1   # number of longitude cells
    Ny = size(bm, 2) - 1   # number of latitude cells
    Nz = size(am, 3)        # number of vertical levels

    # Eigenvalues of the 2D discrete Laplacian on an (Nx × Ny) periodic grid.
    # fac[i, j] = 2(cos(2π(i-1)/Nx) + cos(2π(j-1)/Ny) - 2)
    # fac[1, 1] = 0 for the (0,0) mode (mean); set to 1 to avoid ÷0.
    fac = Array{Float64}(undef, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        fac[i, j] = 2.0 * (cos(2π * (i - 1) / Nx) + cos(2π * (j - 1) / Ny) - 2.0)
    end
    fac[1, 1] = 1.0   # null-space mode — div by 1 then zero explicitly below

    psi = Array{Float64}(undef, Nx, Ny)
    residual = Array{Float64}(undef, Nx, Ny)
    n_balanced = 0
    max_residual = 0.0

    for k in 1:Nz
        # Residual = horizontal convergence − target mass tendency.
        # convergence = (am_west − am_east) + (bm_south − bm_north)
        # (positive convergence = mass accumulating in cell).
        @inbounds for j in 1:Ny, i in 1:Nx
            conv = (Float64(am[i, j, k]) - Float64(am[i + 1, j, k])) +
                   (Float64(bm[i, j, k]) - Float64(bm[i, j + 1, k]))
            residual[i, j] = conv - Float64(dm_dt[i, j, k])
        end

        max_res_k = maximum(abs, residual)
        max_residual = max(max_residual, max_res_k)
        max_res_k < 1e-10 && continue  # already balanced at this level

        # Solve ∇²ψ = residual via FFT division by the Laplacian eigenvalues.
        A = fft(complex.(residual))
        @inbounds for j in 1:Ny, i in 1:Nx
            A[i, j] /= fac[i, j]    # ψ̂ = R̂ / eigenvalue
        end
        A[1, 1] = 0.0 + 0.0im       # zero out the (0,0) mode (global mean)
        psi .= real.(ifft(A))        # streamfunction in physical space

        # Apply zonal flux correction: am[i] += ψ[i] − ψ[i−1] (periodic).
        @inbounds for j in 1:Ny
            u_wrap = psi[1, j] - psi[Nx, j]  # periodic wrap contribution
            for i in 2:Nx
                du = (psi[i, j] - psi[i - 1, j]) - u_wrap
                am[i, j, k] += FT(du)
            end
            am[1, j, k] += FT(0)          # boundary faces: correction wraps
            am[Nx + 1, j, k] += FT(0)     # (already exact via periodicity)
        end

        # Apply meridional flux correction: bm[j] += ψ[j] − ψ[j−1] (periodic lat wrap).
        @inbounds for i in 1:Nx
            v_wrap = psi[i, 1] - psi[i, Ny]  # periodic wrap contribution
            for j in 2:Ny
                dv = (psi[i, j] - psi[i, j - 1]) - v_wrap
                bm[i, j, k] += FT(dv)
            end
        end

        n_balanced += 1
    end

    @info "Poisson balance: corrected $n_balanced/$Nz levels, " *
          "max pre-balance residual: $(round(max_residual, sigdigits=3)) kg"
end

"""
    merge_qv!(qv_merged, qv_native, m_native, mm)

Merge native-level specific humidity to the output vertical grid using native
air mass as the averaging weight.
"""
function merge_qv!(qv_merged::Array{FT, 3}, qv_native::Array{FT, 3},
                   m_native::Array{FT, 3}, mm::Vector{Int}) where FT
    Nx, Ny = size(qv_merged, 1), size(qv_merged, 2)
    Nz_merged = size(qv_merged, 3)
    fill!(qv_merged, zero(FT))
    m_sum = zeros(FT, Nx, Ny, Nz_merged)
    @inbounds for k in 1:length(mm)
        km = mm[k]
        @views qv_merged[:, :, km] .+= qv_native[:, :, k] .* m_native[:, :, k]
        @views m_sum[:, :, km] .+= m_native[:, :, k]
    end
    @inbounds for km in 1:Nz_merged
        @views qv_merged[:, :, km] ./= max.(m_sum[:, :, km], FT(1))
    end
end

"""
    read_qv_from_thermo(thermo_path, hour_idx, Nx, Ny, Nz; FT=Float32)

Read hourly ERA5 model-level specific humidity from the thermo NetCDF sidecar.
The returned field is always oriented `(longitude, south_to_north_latitude, level)`.
"""
function read_qv_from_thermo(thermo_path::String, hour_idx::Int, Nx::Int, Ny::Int, Nz::Int;
                             FT::Type{<:AbstractFloat}=Float32)
    NCDataset(thermo_path) do ds
        q_var = ds["q"]
        dims = dimnames(q_var)
        if dims[1] == "longitude"
            q = FT.(q_var[:, :, :, hour_idx])
        else
            q_raw = FT.(q_var[hour_idx, :, :, :])
            q = permutedims(q_raw, (3, 2, 1))
        end
        if size(q, 2) == Ny && ds["latitude"][1] > ds["latitude"][end]
            q = q[:, end:-1:1, :]
        end
        return q
    end
end

"""
    apply_dry_basis_native!(m, am, bm, qv)

Convert native-level moist mass and horizontal fluxes to dry basis before level
merging. Face humidity is reconstructed by simple two-point averaging.
"""
function apply_dry_basis_native!(m::Array{Float64, 3},
                                 am::Array{Float64, 3},
                                 bm::Array{Float64, 3},
                                 qv::Array{Float64, 3})
    Nx, Ny, Nz = size(m)

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        q = clamp(qv[i, j, k], 0.0, 0.999999)
        m[i, j, k] *= (1.0 - q)
    end

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:(Nx + 1)
        i_l = i == 1 ? Nx : i - 1
        i_r = i <= Nx ? i : 1
        q_face = 0.5 * (qv[i_l, j, k] + qv[i_r, j, k])
        q_face = clamp(q_face, 0.0, 0.999999)
        am[i, j, k] *= (1.0 - q_face)
    end

    @inbounds for k in 1:Nz, j in 1:(Ny + 1), i in 1:Nx
        if j == 1 || j == Ny + 1
            bm[i, j, k] = 0.0
        else
            q_face = 0.5 * (qv[i, j - 1, k] + qv[i, j, k])
            q_face = clamp(q_face, 0.0, 0.999999)
            bm[i, j, k] *= (1.0 - q_face)
        end
    end
    return nothing
end

"""
    recompute_cm_from_divergence!(cm, am, bm, m; B_ifc=Float64[])

Recompute merged-level vertical mass flux from merged horizontal divergence.
When `B_ifc` is supplied, the TM5/ERA5 hybrid-coordinate correction is used.
"""
function recompute_cm_from_divergence!(cm::Array{FT, 3}, am::Array{FT, 3},
                                       bm::Array{FT, 3}, m::Array{FT, 3};
                                       B_ifc::Vector{<:Real}=Float64[]) where FT
    Nx = size(m, 1)
    Ny = size(m, 2)
    Nz = size(m, 3)
    fill!(cm, zero(FT))

    if !isempty(B_ifc) && length(B_ifc) == Nz + 1
        @inbounds for j in 1:Ny, i in 1:Nx
            pit = 0.0
            for k in 1:Nz
                pit += (Float64(am[i + 1, j, k]) - Float64(am[i, j, k])) +
                       (Float64(bm[i, j + 1, k]) - Float64(bm[i, j, k]))
            end
            acc = 0.0
            for k in 1:Nz
                div_h = (Float64(am[i + 1, j, k]) - Float64(am[i, j, k])) +
                        (Float64(bm[i, j + 1, k]) - Float64(bm[i, j, k]))
                acc = acc - div_h + (Float64(B_ifc[k + 1]) - Float64(B_ifc[k])) * pit
                cm[i, j, k + 1] = FT(acc)
            end
        end
    else
        @inbounds for j in 1:Ny, i in 1:Nx
            acc = 0.0
            for k in 1:Nz
                div_h = (Float64(am[i + 1, j, k]) - Float64(am[i, j, k])) +
                        (Float64(bm[i, j + 1, k]) - Float64(bm[i, j, k]))
                acc = acc - div_h
                cm[i, j, k + 1] = FT(acc)
            end
        end
    end
end

"""
    pin_global_mean_ps!(sp, area; target_ps_dry_pa, qv_global) -> Float64

Apply a uniform additive offset to `sp` so that ⟨sp⟩_area corresponds to a
prescribed dry-air mass target inferred from a global humidity climatology.
"""
function pin_global_mean_ps!(sp::AbstractMatrix{Float64},
                             area::AbstractMatrix{Float64};
                             target_ps_dry_pa::Float64 = 98726.0,
                             qv_global::Float64 = 0.00247)
    target_ps_total = target_ps_dry_pa / (1.0 - qv_global)
    sum_ps_area = 0.0
    sum_area = 0.0
    @inbounds for j in axes(sp, 2), i in axes(sp, 1)
        a = area[i, j]
        sum_ps_area += sp[i, j] * a
        sum_area += a
    end
    ps_mean_current = sum_ps_area / sum_area
    offset = target_ps_total - ps_mean_current
    @. sp += offset
    return offset
end

"""
    pin_global_mean_ps_using_qv!(sp, area, dA, dB, qv; target_ps_dry_pa) -> Float64

Apply a uniform additive offset to `sp` so that the area-weighted global mean
dry surface pressure implied by the native-layer hourly humidity field matches
the prescribed target.

The correction is still spatially uniform in `ps`; only the global dry-mass
targeting uses the instantaneous humidity pattern.
"""
function pin_global_mean_ps_using_qv!(sp::AbstractMatrix{Float64},
                                      area::AbstractMatrix{Float64},
                                      dA::AbstractVector,
                                      dB::AbstractVector,
                                      qv::Array{Float64, 3};
                                      target_ps_dry_pa::Float64 = 98726.0)
    Nx, Ny, Nz = size(qv)
    length(dA) == Nz || error("length(dA)=$(length(dA)) does not match qv levels $Nz")
    length(dB) == Nz || error("length(dB)=$(length(dB)) does not match qv levels $Nz")
    size(sp) == (Nx, Ny) || error("sp shape $(size(sp)) does not match qv horizontal shape ($Nx, $Ny)")
    size(area) == (Nx, Ny) || error("area shape $(size(area)) does not match qv horizontal shape ($Nx, $Ny)")

    sum_dry_ps_area = 0.0
    sum_alpha_area = 0.0
    sum_area = 0.0

    @inbounds for j in 1:Ny, i in 1:Nx
        ps_ij = sp[i, j]
        dry_ps_col = 0.0
        alpha_col = 0.0
        for k in 1:Nz
            q = clamp(qv[i, j, k], 0.0, 0.999999)
            dry_ps_col += (Float64(dA[k]) + Float64(dB[k]) * ps_ij) * (1.0 - q)
            alpha_col += Float64(dB[k]) * (1.0 - q)
        end
        a = area[i, j]
        sum_dry_ps_area += dry_ps_col * a
        sum_alpha_area += alpha_col * a
        sum_area += a
    end

    mean_dry_ps_current = sum_dry_ps_area / sum_area
    mean_alpha = sum_alpha_area / sum_area
    mean_alpha > 0.0 || error("Computed non-positive dry-pressure sensitivity: $mean_alpha")

    offset = (target_ps_dry_pa - mean_dry_ps_current) / mean_alpha
    @. sp += offset
    return offset
end
