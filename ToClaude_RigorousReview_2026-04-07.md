# Rigorous Review: Spectral Preprocessing vs TM5 cy3-4dvar

**Audit date**: 2026-04-07
**Auditor**: rigorous-code-reviewer agent (read-only)
**Scope**: `scripts/preprocessing/preprocess_spectral_v4_binary.jl` vs
`deps/tm5-cy3-4dvar/base/src/` spectral preprocessing chain.
**Driving question**: Is the observed -8.31e-6/day drift in `Σm` on
the F64 LL test coming from a bug in our spectral preprocessor?

## Scope of audit

Read in full:
- `scripts/preprocessing/preprocess_spectral_v4_binary.jl` (1461 lines)
- `scripts/preprocessing/preprocess_spectral_massflux.jl` (838 lines; core math is identical to v4)
- `config/era5_L137_coefficients.toml`
- `deps/tm5-cy3-4dvar/base/src/grid_type_sh.F90` (`sh_vod2uv`, `sh_Pnm`, `shgrid_Eval_Lons_Xm_1`)
- `deps/tm5-cy3-4dvar/base/src/grid_interpol.F90` (`IntLat_sh_ll`, `IntLon_sh_ll`, `IntArea_sh_ll_fgh`, `ShRefinement`)
- `deps/tm5-cy3-4dvar/base/src/tmm.F90` (`tmm_Read_MFUV`, `tmm_Read_MFW`)
- `deps/tm5-cy3-4dvar/base/src/advect_tools.F90` (`dynam0`, `m2phlb`)
- `deps/tm5-cy3-4dvar/base/src/tmm_mf_ecmwf_tm5.F90` (spectral → SH packing)
- `deps/tm5-cy3-4dvar/base/src/file_grib.F90` (`grib_Get` spectral path)
- `deps/tm5-cy3-4dvar/base/src/grid_type_hyb.F90` (`levi_FillLevels_3d` `combine`/`interpol` paths)
- `deps/tm5-cy3-4dvar/proj/levels/ml137/tropo34a/src/dims_levels.F90`
- `src/Advection/mass_flux_advection.jl` (runtime `_cm_column_kernel!`, `_massflux_z_kernel!`)
- `src/IO/preprocessed_latlon_driver.jl` (runtime reads cm from disk directly; does not rederive)

## Verified-matching parts (NOT candidates for the drift)

Pieces checked line-by-line and found consistent with TM5:

- **GRIB spectral packing** (v4 line 522-527 vs TM5 `grib_Get` line 1252-1278 + `sh_vod2uv` index order): both pack coefficients iterating `m` outer and `n` inner, starting at `(m=0,n=0)` and ending at `(m=T,n=T)`. Our `spec[n+1, m+1] = complex(vals[idx], vals[idx+1]); idx += 2` matches TM5's `transfer(rsec4, …)` reading of the same GRIB rsec4 block.

- **Fully normalized associated Legendre recurrence** (v4 line 469-498 vs TM5 `sh_Pnm` line 753-833):
  - Sectoral: our `P[m+1,m+1] = sqrt((2m+1)/(2m)) * cos * P[m,m]` equals TM5's `fmm = fmmp * rmu / sqrt(2m)` followed by the `fmmp = fmm * sqrt(2m+3)` update. Derived both for m=1..3 and they agree to symbolic simplification.
  - First off-diagonal: our `P[m+2,m+1] = sqrt(2m+3) * sin * P[m+1,m+1]` equals TM5's `Pnm(k) = fmmp * mu` (TM5 rolls the `sqrt(2m+3)` into `fmmp`).
  - General recurrence: our `a = sqrt((4n²-1)/(n²-m²))` is exactly `1/eps` in TM5's `eps = sqrt((n²-m²)/(4n²-1))`. Our `b = sqrt(((2n+1)(n-m-1)(n+m-1))/((2n-3)(n²-m²)))` equals TM5's `eps1/eps`.

- **`δ` and `σ` definitions**: our `_delta(m,n) = -sqrt((n²-m²)/(4n²-1))/n` and `_sigma(m,n) = -m/(n(n+1))` match TM5's contained `delta` and `sigma` functions exactly (`grid_type_sh.F90:1929-1938`).

- **`vod2uv` core formula**: for the general interior case (`m+1 ≤ n ≤ T-2`), our `U = R*(δ_mn*vo_nm1 + iσ_mn*d_n - δ_mn1*vo_np1)` and `V = R*(-δ_mn*d_nm1 + iσ_mn*vo_n + δ_mn1*d_np1)` match TM5 `grid_type_sh.F90:1903-1904` character-for-character.

- **`vod2uv` m=0 handling**: TM5 explicitly drops the `iσ(0,0)*DI(k)` term at `(m=0,n=0)` because `σ(0,0)=NaN`. Our `_sigma` returns 0 for both `m=0` and `n=0`, so the multiplication contributes nothing. Same behavior.

- **`cm` B-correction formula is self-consistent with the runtime.** The preprocessor's `recompute_cm_from_divergence!` writes `cm[k+1] = -Σ_{k'=1..k} div_h[k'] + B_ifc[k+1]*pit` (lines 373-377 of v4 and 582-583 of older script). The runtime `_cm_column_kernel!` at `src/Advection/mass_flux_advection.jl:64-80` uses the identical recurrence `acc += conv_k - bt[k]*pit` with `bt[k] = (B[k+1]-B[k]) / (B[Nz+1]-B[1])`. Since `B[Nz+1]-B[1] = 1` for full columns, `bt[k] = dB[k]`, so the two are byte-identical. Both give `cm[1]=0`, `cm[Nz+1]=0` by construction.

- **`cm` sign convention matches runtime z-advection.** The runtime `_massflux_z_kernel!` at `src/Advection/mass_flux_advection.jl:388-419` treats positive `cm[k]` as DOWNWARD flux (donor is `m[k-1]` = layer above). Tracing through our formula for the all-outflow test case (`div_h[k]=δ, pit=Nz*δ`) gives `cm[k+1] = -k*δ + B[k+1]*Nz*δ`, which is positive near the surface (where `B` is close to 1) and slightly negative near TOA — consistent with an outflowing column producing downward cm where the pressure surfaces are squeezing down.

- **Equivalence to TM5 `dynam0` up to sign convention**. Mapped TM5's bottom-up indexing (`l=1` surface, `lmr+1` TOA, `bt(1)=1, bt(lmr+1)=0`) to our top-down indexing (`k=1` TOA, `Nz+1` surface, `B_ifc[1]=0, B_ifc[Nz+1]=1`) via `k = Nz+1-l` and `B_ifc[k] = bt(Nz+2-k)`. Under that mapping, `dB[k] = bt(l+1)-bt(l+2)` matches TM5's `(bt(l+1)-bt(l+2))*pit` term. TM5's `pit_TM5 = Σ conv_adv` is `-pit_ours` (TM5 uses inflow, we use outflow), so the signs work out. TM5's `cm_TM5` is divided through `-dtw`; ours has `half_dt` already baked into `am/bm`.

- **Level merging for `m`, `am`, `bm`** (`merge_cell_field!`): summing native layers into merged layers is exactly equivalent to computing on merged levels for `m` (because `Σ (dA_native + dB_native*ps) = dA_merged + dB_merged*ps` by construction of the merged dA/dB) and for horizontal fluxes because `Σ_native u_staggered * dp_native = u_staggered * dp_merged` (u does not depend on native level in our code since we only store one U/V per layer and they are not re-interpolated during merging). See Finding M1 for the cm handling difference.

- **ECMWF L137 orientation**: `b[0]=0, b[137]=1` in `config/era5_L137_coefficients.toml` matches the "n=0 top, n=137 surface" header comment and matches TM5's use via `echlevs(0)=137, echlevs(lm)=0` + `bt(1:lm+1) = b_ec(echlevs)`.

## Findings

### H1 — Spectral quadrature of mass fluxes is structurally different from TM5 (high severity, likely NOT the drift source)

**Where**: v4 `compute_mass_fluxes!` at lines 752-807, older script at lines 519-588.
**TM5 reference**: `tmm.F90:2136-2147` (the actual mfu formula) and `grid_interpol.F90:3127-3309` (`IntLat_sh_ll`).

**TM5's mfu**:
```fortran
! tmm.F90:2136-2139
call IntLat( '(da+exp*db)/cos', shi, nlev, U_sh, LNSP_sh%c, &
             tmm%buf_levi%da, tmm%buf_levi%db, lli, mfuX, status )
mfuX = mfuX * R/g
```
Inside `IntLat_sh_ll` (line 3208-3292) TM5:
1. Iterates over `j = 1..jm` cell rows.
2. For each row, loops over `jf = 0..refinement_j` fine sub-latitudes (refinement_j = 2 for T=359, dlat=0.5°, via `ShRefinement` at line 1593-1621).
3. At each fine latitude, **evaluates U spectrally at `lli%blon(0)` as the starting longitude** via `Eval_Lons(..., lli%blon(0), nlon_fine+1, ...)`. This gives U at longitudes `blon(0), blon(0)+dlon, ...` — i.e., directly at the cell FACES.
4. Also spectrally evaluates `exp(LNSP)` at the same fine latitudes/longitudes.
5. Forms `llf(i, jf, l) = ff * (da(l) + exp_hh*db(l)) / cos(lat(jf))` — integrand evaluated pointwise at the exact face longitudes and fine latitudes.
6. Integrates `llf(i, :, l)` in latitude from `blat(j-1)` to `blat(j)` via `IntervalQuad_Lin` (piecewise-linear trapezoidal).
7. Multiplies by `R/g` to get `mfu` in kg/s (NOT per half-timestep, that's a separate scale).

So TM5's `mfu(i, j, l)` is approximately `∫_{blat(j-1)}^{blat(j)} U(blon(i), lat) * (da(l) + db(l)*exp(LNSP(blon(i), lat))) / cos(lat) dlat * R/g`.

**Our mfu/am** (v4 line 763-769):
```julia
@inbounds for k in 1:Nz, j in 1:Nlat, i in 1:(Nlon+1)
    i_l = i == 1 ? Nlon : i - 1
    i_r = i <= Nlon ? i : 1
    dp_stag = (dp[i_l, j, k] + dp[i_r, j, k]) / 2
    cos_lat = max(grid.cos_lat[j], 1e-10)
    am[i, j, k] = u_stag[i, j, k] / cos_lat * dp_stag * R_g * dlat * half_dt
end
```

Ours is:
1. `u_stag[i, j, k]` is `(u_cc[i-1, j, k] + u_cc[i, j, k])/2` (from `stagger_winds!`, v4 line 708-720). `u_cc` was synthesized from spectral `U_sh` via `spectral_to_grid!`, which evaluates at longitudes `lon_k = (k-1) * 360/Nlon = 0, 0.5°, 1°, ..., 359.5°` (BFFT default sample positions).
2. `dp_stag` is a simple average of the two neighboring cell-center `dp` values. Each cell-center `dp` is `|dA + dB*ps|` where `ps = exp(LNSP on grid)` — so `dp_stag` is NOT `da + db*exp(LNSP at face)`. Because `exp` is nonlinear, `(exp(x1)+exp(x2))/2 ≠ exp((x1+x2)/2)`.
3. `dlat` is the constant cell width in radians, not an integral.
4. One quadrature point at cell center `lats[j]` per cell row; TM5 uses `refinement_j + 1 = 3` points per row at T=359.

These differ at three levels:
- **Quadrature order**: our midpoint rule vs TM5's 2-interval trapezoidal. Both are O(Δφ²) accurate but differ by a level-dependent constant.
- **Pressure-thickness averaging**: we average the nonlinear `exp(LNSP)` after evaluating at two cell centers. TM5 averages `exp(LNSP)` after evaluating on a fine lat/lon grid that lands ON the cell face longitudes. Our averaged `dp_stag` has a second-order Jensen bias `+ (var_ij(LNSP)/2) * exp(LNSP_avg) * db` per cell.
- **Sample location**: `u_stag[i, j, k]` represents U at the **cell center longitude** (not the west face) because we average BFFT outputs at `0°, 0.5°, ...` to get values at `0.25°, 0.75°, ...`. But we use this value in the flux expression labeled "at west face i". TM5 evaluates U AT the face directly (`blon(0) = 0°`).

**Does this cause the 8e-6/day global drift?**

Argument for "no":
- Our global mass (both at a single window and summed across windows) is driven only by the global mean of `ps = exp(LNSP)` on our 720×361 grid. That's `Σ_ij area_ij * exp(field_2d[i,j])` where `field_2d` is the truncated-T359 spectral synthesis of LNSP. This is computed in a single path (v4 line 851-852) and does NOT go through `compute_mass_fluxes!`. The am/bm quadrature artefact does not affect `m[i,j,k]` at all — only the horizontal flux field which is overridden by the Poisson balance anyway.
- After the Poisson balance (v4 line 1184), `conv(am_new, bm_new) = dm_dt - mean(dm_dt)` per cell (where `dm_dt = m[win+1] - m[win]`, a linear function of `ps[win+1] - ps[win]`). So any am/bm bias is nulled by the balance as long as `dm_dt` is computed correctly, which it is.

Argument for "maybe":
- The quadrature bias changes per window, and its zero-mean projection is not nulled by the balance (only its mean is absorbed). But this affects the spatial pattern of advection, not the global mass.

**Best read: H1 is a real algorithmic divergence from TM5, worth documenting in project invariants as "not-TM5-faithful midpoint-rule quadrature", but it is NOT the source of the drift because the drift lives in `m` via `ps`, not in `am/bm`.**

### H2 — `u_cc` and `v_cc` are sampled at cell faces but treated as cell centers (high severity, NOT the drift source)

**Where**: v4 `spectral_to_grid!` at lines 643-693, `stagger_winds!` at lines 708-720, and `TargetGrid` constructor at lines 407-437.

**Evidence**:
- `TargetGrid(Nlon, Nlat)` (line 417) sets `lons = [dlon_deg * (i - 0.5) for i in 1:Nlon]`. For Nlon=720, `lons[1] = 0.25°, lons[2] = 0.75°, …`.
- `spectral_to_grid!` (line 688-690) writes `field[i, j] = real(f_lon[i])` where `f_lon = bfft(fft_buf)`. For a real-input-spectrum bfft, `bfft(G)[k] = Σ_m G[m] exp(+2πi m (k-1)/N) = f(λ = 2π(k-1)/N)`. In degrees this is `λ_1 = 0°, λ_2 = 0.5°, …`. **These are NOT the cell centers declared in `TargetGrid`.** They are the cell WEST FACES (`blon(i) = (i-1) * dlon_deg`).
- `stagger_winds!` (line 708-713) then computes `u_stag[i,j,k] = (u_cc[i,j,k] + u_cc[ip,j,k])/2`. Interpreting `u_cc[i]` at its true location `(i-1)*dlon`, the averaged value is at `(i-0.5)*dlon` = the cell CENTER (`lons[i]`). But we then use this as the flux at the WEST FACE of cell `i` in `compute_mass_fluxes!`.

So there is an end-to-end half-cell spatial shift: `u_cc` lives at faces, gets averaged to centers, is then labeled as living at faces again.

**TM5 reference**: `tmm.F90:2136` passes `lli%blon(0)` as the `lon_start` argument to `Eval_Lons`, which (per `shgrid_Eval_Lons_Xm_1` line 1679-1681: `X(m) = Xm(m) * exp((0,1)*m*lon_start)`) shifts the FFT so that sample `k=0` lands at `blon(0)` = the western-most CELL FACE. TM5 then evaluates U on a fine grid starting at that face. **TM5 evaluates U AT faces directly**, no post-hoc staggering of cell-center values.

**Does it cause the drift?**
- Global-mass impact is zero: `Σ_ij (am[i+1] - am[i])` telescopes to zero over any periodic row regardless of the spatial offset, because the function `am(i)` is evaluated on a uniform grid and the discrete difference still forms a complete telescoping sum per row.
- Per-cell divergence pattern has a consistent half-cell bias, but this feeds into the Poisson balance which absorbs the mean and re-derives cm. With mass_fixer=true the runtime imposes the binary's dm anyway, so the local pattern bias is washed out.
- Confidence that this does not explain 8e-6/day: medium-high. The mechanism by which a spatially-shifted am/bm pattern could produce a GLOBAL mass drift is unclear and no scenario constructs plausibly.

**Why this is still worth reporting**: it is a real algorithmic discrepancy from TM5 that may matter for other test cases (e.g., comparison of local tracer fields with TM5 runs using the same ERA5 inputs, non-uniform IC tests). It's also the kind of off-by-half-cell bug that could surface in future as a "why doesn't our result match TM5 locally" question.

**Mitigation**: either
  (a) set `lons[i] = (i-1) * dlon_deg` instead of `(i-0.5) * dlon_deg` (makes the label match the BFFT output), and change the mass-flux face-loop accordingly so `am[i, j, k]` is the flux at `lons[i] - dlon_deg/2 = (i-1.5)*dlon_deg`; OR
  (b) shift the spectral coefficient by `exp(i*m*dlon/2)` before BFFT to get values at `(k-0.5)*dlon` = true cell centers, then stagger to faces at `(k-1)*dlon = lons[i] - dlon_deg/2`.

### H3 — Our truncation at `T = Nlon/2 - 1` drops information from T639 spectral files (low, bias is constant per window and does not drift)

**Where**: v4 `main` at line 1375: `T_target = div(Nlon, 2) - 1  # Nyquist: 720/2 - 1 = 359`. Then `read_day_spectral_streaming` at line 542-553 slices the spectral array down to `T+1` rows and columns, dropping the higher m and n modes by simply not copying them.

**TM5 reference**: `IntLat_sh_ll:3183` calls `T = ShTruncation( T, lli%dlon_deg, lli%dlat_deg )` which gives approximately the same `T = ceil((360/dlon_deg - 1)/2) = 359` for our grid. But TM5's handling of the truncation is to initialize a new `sh` grid of lower truncation and use it consistently in both the Legendre evaluation and the summation. Our code uses the original `T+1 × T+1` spec array with just the top-left block nonzero — same mathematical effect.

**Does it drift?** Because `exp(LNSP)` is nonlinear, truncating LNSP to T=359 produces a constant relative bias in `<ps>` (from dropping `LNSP_high`, variance of which acts as a second-order contribution via `<exp(LNSP_low) * (LNSP_high²/2)>`). But this bias is approximately the same every window (assuming the variance of the dropped high-frequency LNSP is stationary over 24h, which it should be for sub-1° topographic + weather features). So it contributes to the absolute level of `<ps>` but not to temporal drift.

**No action.** Just noting that this introduces a small absolute (~1e-5 relative) bias in `<ps>` vs running at the full T639, which is not a drift source.

### H4 — The `sh_vod2uv` truncation boundary handling differs subtly between TM5 and us at `n = T-1` (low, no drift effect, ours is mathematically cleaner than TM5)

**Where**: v4 `vod2uv!` at line 605-630 vs TM5 `sh_vod2uv` at `grid_type_sh.F90:1886-1920`.

**TM5 at `n = T-1` (line 1908-1911)**:
```fortran
! (m,T-1)
n = T-1
k  = k + 1
U%c(k) = (   delta(m,n)*VO%c(k-1) + z*sigma(m,n)*DI%c(k))*R
V%c(k) = ( - delta(m,n)*DI%c(k-1) + z*sigma(m,n)*VO%c(k))*R
```
TM5 omits the `-delta(m,n+1)*VO(k+1) = -delta(m,T)*VO(m,T)` term, even though `VO(m,T)` is a valid coefficient stored in the truncated array.

**Us at `n = T-1`** (v4 line 619-626): our loop iterates `for n in m:T`. At `n = T-1`, the condition `n < T` is true, so `vo_np1 = vo_spec[n+2, m+1] = vo_spec[T+1, m+1] = VO(m, T)`. We DO include the `-delta(m,T)*VO(m,T)` term.

This is a boundary-truncation discrepancy. Our code is mathematically more correct (it uses all available spectral information). TM5 drops a contribution that would have been valid.

**Drift impact**: zero. This changes the grid-point U/V by a small amount in a spatially-patterned way; it does not bias the global mean of either U or ps. And since our global mass only depends on ps, there is no plausible drift source here.

**No action**; noted for completeness.

### H5 — Pole-row U garbage (`u_stag/cos(φ) ~ O(1e17)`) in our v4 is a SYMPTOM of the cos-factor convention difference, not a distinct bug (medium, does not drift)

**Where**: v4 `compute_mass_fluxes!` line 767-768:
```julia
cos_lat = max(grid.cos_lat[j], 1e-10)
am[i, j, k] = u_stag[i, j, k] / cos_lat * dp_stag * R_g * dlat * half_dt
```
plus the runtime pole-row am zeroing in `physics_phases.jl` noted in `FromClaude.md` open question 6.

**TM5 reference**: `IntLat_sh_ll:3263`:
```fortran
llf(:,jf,l) = ff * ( da(l) + exp_hh*db(l) ) / cos(lat(jf))
```
TM5 also divides by `cos(lat)`, but line 3212-3229 adds special handling for the poles: if the fine latitude `lat(jf)` is within `1e-4` of ±π/2, TM5 uses the AVERAGE of the next/previous row rather than dividing by near-zero cos(lat).

**Our code** does not have this pole special case. Instead it clamps `cos_lat >= 1e-10`, which lets `am[:, j=1, :]` and `am[:, j=Ny, :]` blow up to numerically gigantic values. `FromClaude.md` notes `max abs ~ 4.55e17` is "defensive cleanup", but this is only true because the `(am[i+1]-am[i])` reduction cancels the garbage to exactly zero at the pole rows (uniform-in-i garbage has zero longitudinal divergence). TM5 avoids this pathology entirely by using row-averaging instead of dividing by cos(lat) at the pole rows.

**Drift impact**: none for mass conservation (the cancellation in `am[i+1]-am[i]` is exact at machine precision for the uniform garbage). But the Poisson balance step SEES this garbage as input because it computes `conv(am, bm)` which is the sum over all i. The mean and variance of the column residual at j=1/Ny will be dominated by Float32 roundoff of `~1e17` minus `~1e17`. **This could plausibly inject spatially-white noise into the Poisson solution at pole cells** even though the true signal is zero.

**Recommendation**: either adopt TM5's pole-row averaging, or explicitly zero `am[:, 1, :]` and `am[:, Ny, :]` in `compute_mass_fluxes!` BEFORE the Poisson balance step runs (currently the zeroing happens in the runtime, but the preprocessor Poisson solve sees the garbage).

### H6 — Poisson balance's zero-mean-mode discard silently converts global mass imbalance to per-cell flux residual (medium, may not be the drift source but is architecturally load-bearing)

**Where**: v4 `balance_mass_fluxes!` at lines 220-295, specifically line 231 (`fac[1,1] = 1.0`) and line 257 (`A[1,1] = 0.0`).

**Math**: the discrete 2D Laplacian on a periodic `Nx × Ny` grid has a null space of one dimension: the constant mode. The Poisson equation `∇²ψ = residual` has a solution only if `mean(residual) = 0`. By setting `A[1,1] = 0`, the code is computing a solution to `∇²ψ = residual - mean(residual)` (the mean-zero projection), and discarding the non-zero-mean part.

After balancing:
```
conv(am_new, bm_new) = conv(am, bm) - ∇²ψ
                     = (dm_dt + residual) - (residual - mean(residual))
                     = dm_dt + mean(residual)
```
Because `Σ_ij conv(am, bm) = 0` identically (telescoping on periodic x, pole-zero bm on y), `mean(residual) = -mean(dm_dt)`. So after balance, `conv(am_new, bm_new) = dm_dt - mean(dm_dt)` per cell.

**At the global level**: the per-cell residual `-mean(dm_dt)` is a uniform constant. Summed globally, `Σ [dm_dt - mean(dm_dt)] = 0`. So the am/bm is globally mass-conserving (satisfies `Σ conv = 0`).

**At the per-cell level**: the advection run (without mass_fixer) would move mass per the corrected am/bm, giving `m_new_advected[i] = m[i] + (dm_dt[i] - mean(dm_dt))`. This is NOT equal to the binary's stored `m_next[i] = m[i] + dm_dt[i]`. The discrepancy at each cell is exactly `mean(dm_dt)` (uniform).

**With mass_fixer=true**: the runtime overrides `m` to match the binary's dm trajectory at each substep. So the `mean(dm_dt)` offset is absorbed by mass_fixer, and tracers are rescaled. This is how the current tests run without crashing.

**Is this the drift mechanism?** No. The drift is in the binary's `dm` field directly. The Poisson solve doesn't change the `dm` field — it only modifies `am/bm/cm`. The `dm` written to disk is `all_m[win+1] - all_m[win]` at lines 1219-1232, which is computed BEFORE the Poisson step (and the Poisson step wouldn't affect it even if the Poisson step DID touch m, which it doesn't — `all_m` is read-only in `balance_mass_fluxes!`).

**The drift in `Σdm` is fully explained by the global mean of `ps = exp(LNSP)` changing from window 1 to window 24.** This is a property of the ERA5 input LNSP, not of our Poisson balance.

**Open**: whether TM5 has an equivalent operation. TM5's `tmm_Read_MFW` computes `tsp = -IIOmega(:,:,nlev)` (surface pressure tendency from spectral vertical flux integral) and then uses this to derive the target `ps_new`. There is no "Poisson balance of am/bm against stored dm" step — TM5 advection runs with just am, bm, and the dynamically-derived cm from `dynam0`, and `m` follows from `(phlb(l) - phlb(l+1))*dxyp/g` at each substep (see `advect_tools.F90:100-130` for `m2phlb1`).

**So TM5 has NO analog of our Poisson balance.** Our balance is a project addition, not a TM5-faithful operation. The CLAUDE.md comment "TM5 r1112 effectively does mass-fixing via m = (at + bt × ps) × area / g each substep" (invariant 11) is closer to TM5's behavior — but TM5 doesn't apply a Poisson solve to force am/bm to match that m trajectory. TM5 just reconstructs m from ps each substep and lets advection drift within the window (bounded by Check_CFL refinement).

**Recommendation**: document in CLAUDE.md that `balance_mass_fluxes!` is a project-specific additional step, not present in TM5 r1112. The Poisson step + mass_fixer together form an architecturally different approach to mass closure than TM5's re-derivation from ps. Whether one approach is "more correct" is a design question, not a bug question.

### M1 — `merge_cell_field!` sums native levels; TM5 `FillLevels` with key=`sum` does the same for "combine" reverse mapping (medium, matches for m/am/bm, but cm handling differs)

**Where**: v4 `merge_cell_field!` at lines 195-202 vs TM5 `levi_FillLevels_3d` at `grid_type_hyb.F90:2058-2106`.

**TM5**:
```fortran
case ( 'combine' )
  ll = 0.0
  do l = 1, levi%nlev
    le1 = levi%flevs(l,1); le2 = levi%flevs(l,2)
    do le = le1, le2
      k = le
      if ( reverse ) k = levi%nlev_parent + 1 - le
      select case ( combine_key )
        case ( 'sum' )
          ll(:,:,l) = ll(:,:,l) + llX(:,:,k)
```
This is exactly our `merge_cell_field!` — sum native levels into the merged level.

For vertical flux mfw (half-level data), TM5 uses the `'w' / combine / 'bottom'` path (line 2274-2278):
```fortran
do l = 0, levi%nlev
  k = levi%hlevs(l)
  if ( reverse ) k = levi%nlev_parent - k
  ll(:,:,l+1) = llX(:,:,k+1)
end do
```
TM5 PICKS the native cm at the half-level index that corresponds to the merged interface (`levi%hlevs(l)` = parent interface index), without summing or re-deriving.

**Our v4 approach** (lines 1103-1118):
1. `merge_cell_field!(am_merged, am_native, merge_map)` — sum am native → merged. ✓ matches TM5.
2. `merge_cell_field!(bm_merged, bm_native, merge_map)` — sum bm native → merged. ✓ matches TM5.
3. `recompute_cm_from_divergence!(cm_merged, am_merged, bm_merged, ...)` — **RE-DERIVE cm from the merged am/bm via the B-correction formula.** This is NOT what TM5 does; TM5 picks the native cm at the merged interface.

**Are these equivalent?** For the specific case where the merged am/bm are exact sums of native am/bm, yes — both methods give the same answer up to floating-point roundoff. Proof:
- TM5 at merged interface `k_merged` = native interface `k_native`:
  `cm_TM5_merged(k_merged) = cm_TM5_native(k_native) = -Σ_{l=1..k_native-1} conv_TM5(l) + bt_native(k_native)*pit_TM5`
- Ours: `cm_merged(k_merged+1) = -Σ_{k'=1..k_merged} div_h_merged(k') + B_ifc_merged(k_merged+1)*pit_merged`
- `div_h_merged(k') = Σ_{l in native group k'} div_h_native(l)`, so `Σ_{k'=1..k_merged} div_h_merged(k') = Σ_{l=1..k_native_below} div_h_native(l)`.
- `pit_merged = Σ div_h_merged = Σ div_h_native = pit_native`.
- `B_ifc_merged(k_merged+1) = B_ifc_native(k_native_below+1)`.

Up to the index offset of one (our cm indexing vs TM5's), these are identical.

**Recommendation**: no action needed for drift investigation. Both methods give equivalent results.

### M2 — `merge_cell_field!` thread safety / aliasing (low, cosmetic)

**Where**: v4 `merge_cell_field!` at lines 195-202:
```julia
function merge_cell_field!(merged::Array{FT,3}, native::Array{FT,3}, mm::Vector{Int}) where FT
    Threads.@threads for km in 1:size(merged, 3)
        @views merged[:, :, km] .= zero(FT)
    end
    @inbounds for k in 1:length(mm)
        @views merged[:, :, mm[k]] .+= native[:, :, k]
    end
end
```

The zeroing loop is threaded over merged levels, but the summation loop is serial and iterates over native levels `k = 1..length(mm)`. This is correct and thread-safe. **No issue here.**

### M3 — `_merge_levels_latlon!` in binary reader does NOT use the B-correction, while the preprocessor does (medium, irrelevant for v4 binary path but divergence from preprocessor in NetCDF path)

**Where**: `src/IO/preprocessed_latlon_driver.jl:728-761`. This function is called when the runtime loads a NetCDF file with a merge_map (not a binary file). The cm recomputation here uses:
```julia
div_h = (cpu_buf.am[i+1, j, k] - cpu_buf.am[i, j, k]) +
        (cpu_buf.bm[i, j+1, k] - cpu_buf.bm[i, j, k])
cpu_buf.cm[i, j, k+1] = cpu_buf.cm[i, j, k] - div_h
```
This is the plain cm reconstruction WITHOUT the B-correction term, meaning `cm[Nz+1]` will in general NOT equal zero at each cell.

The preprocessor v4 `recompute_cm_from_divergence!` DOES include the B-correction `+ dB[k]*pit`, giving `cm[Nz+1] = 0` per cell by construction.

**Drift relevance**: None for the v4 binary path (line 677-685 of the driver reads cm directly from disk, does not call `_merge_levels_latlon!`). This divergence only matters if the runtime is fed a NetCDF-merged file. For the F64 debug test we're investigating, we use the v4 binary path. **No impact on the observed 8e-6/day drift.**

**Recommendation**: make `_merge_levels_latlon!` use the same B-correction formula as the preprocessor for consistency. This is a separate issue from the drift.

### M4 — `compute_mass_fluxes!` uses `dlat` (a scalar) and `cos_lat` (from cell centers) where TM5 integrates over the cell; this produces a latitude-dependent second-order error that doesn't null out (medium, not the drift source)

**Where**: v4 line 767-768.

**Math of the discrepancy**: TM5's IntLat produces
`mfu_TM5(i, j, l) = (R/g) * ∫_{blat(j-1)}^{blat(j)} U(blon(i), lat) * (da(l) + db(l)*exp(LNSP(blon(i), lat))) / cos(lat) dlat`
using trapezoidal quadrature on a fine grid.

Our `am(i, j, k) = u_stag * dp_stag * R_g * dlat / cos_lat * half_dt`
is approximately
`(R/g) * U(lons(i), lats(j)) * (da(k) + db(k)*exp(LNSP(lons(i), lats(j)))) / cos(lats(j)) * (blat(j) - blat(j-1))`
where `(blat(j) - blat(j-1)) = dlat` — a midpoint rule with ONE sample at the cell center.

The exact integrand is `F(lat) = U(lat) * (da + db*exp(LNSP(lat))) / cos(lat)`. For the midpoint rule:
`∫ F dlat ≈ F(lat_center) * dlat`.

The error is `(dlat²/24) * F''(lat_center) + O(dlat^4)`. TM5's trapezoidal rule over 2 intervals gives error `-(dlat²/12) * F''(...)/2 = -(dlat²/24) * F''(...)`.

**So midpoint and trapezoidal-2-interval differ by a factor of 2 in the error constant, same sign for the midpoint and opposite sign for the trapezoidal.** Both are O(dlat²) → at dlat=0.5° this is a relative error of roughly `(π/360)²/24 ~ 3e-6` times the second derivative of F.

`F''(lat)` is dominated by the `1/cos(lat)` factor at high latitudes. Near the poles, `1/cos(lat)` diverges as `~1/(π/2 - lat)`, so `F''` is very large near the poles. This means midpoint vs trapezoidal quadrature gives different answers for polar mfu flux by a factor that grows with latitude.

**Is this the drift source?** Likely not. The error is not a drift: it's a persistent spatial bias in am/bm that is the SAME structure every window. For a stationary flow, the bias would cause tracer mis-transport but not global mass drift (globally, `Σ am[i+1] - am[i]` telescopes to zero regardless of the per-cell error).

**But**: this IS a reason our runs might differ from a TM5 reference run on the same ERA5 data. The polar regions particularly would see different advection behavior.

**No action for drift investigation.** This is an algorithmic improvement opportunity.

### M5 — Consistency between stored `dm` and Poisson-balanced `am/bm` (low, cross-check, no bug)

**Where**: v4 line 1174-1193 (Poisson balance loop) and line 1205-1237 (write loop).

The Poisson balance loop updates `all_am[win_idx]`, `all_bm[win_idx]` in place. Then it recomputes `all_cm[win_idx]` from the balanced am/bm. It does NOT touch `all_m[win_idx]`.

Then the write loop at line 1217-1232 computes:
```julia
dm_merged .= all_m[win_idx + 1] .- all_m[win_idx]
```
using the UN-modified `all_m` values. So the `dm` written to disk is the pure mass difference, independent of the Poisson balance.

**This is consistent**: `all_m[win]` represents the target mass at window `win`, derived from `ps[win]` via the A/B formula. The Poisson balance adjusts the fluxes to match `dm_dt = all_m[next] - all_m[curr]` at each cell (minus the mean), but stores both m and the originally-computed dm. With mass_fixer=true the runtime uses dm to set m each substep. With mass_fixer=false the runtime uses conv(am, bm) = dm - mean(dm) to advect m.

**No bug, but a subtle consistency**: if a future change to the write loop accidentally re-derives dm from the balanced am/bm via `conv(am_new, bm_new)`, the dm would be off by `mean(dm)` per cell (a uniform offset that would then be absorbed by mass_fixer silently). Current code is correct; just worth keeping in mind.

### L1 — `ShRefinement(T=359, dlon=0.5°) = 2` means TM5 uses 3 sample points per cell row (low, not a bug)

Noted for completeness. `grid_interpol.F90:1617-1619` sets `shres = 360/(2*(T+1)) = 0.5°` and then `nstep=2`, so `refinement = max(1, ceiling(2*0.5/0.5)) = 2`. 2 sub-intervals per cell = 3 sample points, trapezoidal over 2 intervals. Our midpoint rule uses 1 sample point. This is the quadrature discrepancy discussed in H1 and M4.

### L2 — No equivalent of TM5's `Nabla(LNSP)` term in our cm computation (low, path differs)

**Where**: TM5 `tmm_Read_MFW` at line 2443-2458:
```fortran
call Nabla( LNSP_sh, NabLNSP_sh )
call IntArea( 'F*G*(db*exp(H))/cos', ..., U_sh, NabLNSP_sh(1)%c, LNSP_sh%c, ..., IIOmega2 )
call IntArea( 'F*G*(db*exp(H))/cos', ..., V_sh, NabLNSP_sh(2)%c, LNSP_sh%c, ..., IIOmega2 )
```

TM5 computes the vertical flux from TWO spectral integrals:
1. `IIOmega = ∫∫ D * (da + db*exp(LNSP)) * cos(lat) dA` — divergence × pressure thickness integrated over cell.
2. `IIOmega2 = ∫∫ (U*∂_x(LNSP) + V*∂_y(LNSP)) * db*exp(LNSP)/cos(lat) dA` — pressure-gradient cross-term for hybrid coordinates.

Then combines them: `IIOmega += R² * IIOmega / g + R * IIOmega2 / g`, cumulates from top, and derives `mfwX(l+1) = IIOmega(nlev)*b(l) - IIOmega(l)`.

**Our approach** (v4 `recompute_cm_from_divergence!` at line 353-391) computes cm from the continuity formula applied to our gridpoint am/bm. The two `IIOmega` and `IIOmega2` terms are implicitly combined because we build am/bm from `u_stag * dp_stag / cos` and `v_stag * dp_stag`, whose horizontal difference includes the pressure-gradient term.

**Mathematical equivalence**: TM5's two-integral decomposition and our single horizontal-divergence approach both derive the vertical flux from the continuity equation. They are equivalent up to quadrature error. Ours has the H1 midpoint-rule error; TM5's has its own quadrature error (3-point trapezoidal on a fine grid).

**Drift impact**: none beyond H1/M4.

### L3 — `compute_dp!` uses `abs(dA[k] + dB[k]*ps)` (low, defensive)

v4 line 727-732:
```julia
function compute_dp!(dp, ps, dA, dB, Nlon, Nlat, Nz)
    @inbounds for k in 1:Nz, j in 1:Nlat, i in 1:Nlon
        dp[i, j, k] = abs(dA[k] + dB[k] * ps[i, j])
    end
    return nothing
end
```
For ECMWF L137 with `dA ≥ 0` and `dB ≥ 0` and `ps > 0`, the quantity `dA + dB*ps` is always positive. The `abs` is defensive. TM5 `grid_type_hyb.F90:2086-2087` uses the same `abs`. **No issue.**

## Assumptions Audit

| Assumption | Evidence | Status |
|------------|----------|--------|
| GRIB spectral packing: `m` outer, `n = m..T` inner, complex pairs | TM5 `grib_Get` line 1274-1277 uses `transfer(rsec4, …)` on the same byte order. Our `spec[n+1, m+1] = complex(vals[idx], vals[idx+1]); idx += 2` matches | Verified |
| Fully-normalized associated Legendre polynomials via standard recurrence | Our recurrence coefficients `a`, `b` match TM5's `1/eps`, `eps1/eps` after algebraic simplification | Verified |
| `vod2uv` formula using `delta` and `sigma` matches ECMWF/TM5 convention | Line-by-line comparison with TM5 `sh_vod2uv` interior case | Verified (with small truncation discrepancy at n=T-1, flagged as H4) |
| ECMWF L137 orientation: `b[0]=0` at TOA, `b[137]=1` at surface | `config/era5_L137_coefficients.toml:59-87` | Verified |
| TM5 uses `echlevs(0)=137, echlevs(lm)=0` so level 1 = surface (bottom-up) | `deps/tm5-cy3-4dvar/proj/levels/ml137/tropo34a/src/dims_levels.F90:39-48` | Verified |
| Our level 1 = TOA (top-down), `B_ifc[1]=0, B_ifc[Nz+1]=1` | `load_ab_coefficients` at v4:442-461 takes slice `a_all[i_start:i_end]` of the TOA-to-surface array | Verified |
| Runtime `_cm_column_kernel!` uses same formula as preprocessor | `src/Advection/mass_flux_advection.jl:64-80` vs v4 preprocessor | Verified |
| Preprocessor's cm sign matches runtime z-advection's expectation | Runtime `_massflux_z_kernel!` at line 330-421 uses `cm[k] > 0` as downward; our formula gives positive cm near the surface for outflow | Verified (by hand derivation for uniform-outflow test case) |
| `Σ_ij conv(am, bm) = 0` structurally | Periodic am in i, zero bm at poles | Verified |
| Binary reader reads cm directly (no rederivation) | `src/IO/preprocessed_latlon_driver.jl:677-685` | Verified |
| Poisson balance discards only the mean mode, preserving global mass conservation | Hand derivation in finding H6 | Verified |
| LNSP truncation from T_file to T_target is handled correctly | `read_day_spectral_streaming` at line 547-553 slices `spec_buf[1:T+1, 1:T+1]` correctly | Verified |
| `dm` written to disk = `m[next] - m[curr]` before Poisson balance | v4 line 1219-1226 uses raw `all_m` | Verified |
| TM5's `tmm_Read_MFW` computes vertical flux via two separate spectral integrals and cumulates from top | TM5 line 2419-2537 | Verified |
| TM5 has NO Poisson balance of am/bm against stored dm | Grep for `Nabla`, `SolvePoissonEq` in `deps/tm5-cy3-4dvar/base/src/` shows only the zoom-boundary Poisson solver in grid_type_ll, not used in the main flux path | Verified |
| 8.3e-6/day drift in Σdm is entirely determined by the spatial structure of LNSP (via `<exp(LNSP)>`), not by the preprocessor | Cannot directly verify without running a diagnostic on LNSP at each window. FromClaude.md cites `/tmp/probe_dry_mass_residuals.jl` showing `Σdm/Σm` matches runtime tracer drift exactly | **Plausible but unproven** |
| Our `bfft`-based `spectral_to_grid!` evaluates at longitudes `(k-1)*dlon` rather than `(k-0.5)*dlon` | Definition of `bfft`, `bfft(G)[k] = Σ_m G[m] exp(+2πi m (k-1)/N)`, first output at `λ=0` | Verified |
| The half-cell longitudinal offset between BFFT output and `lons` labels does not break global mass conservation | Telescoping argument for periodic rows, which is independent of the sample positions | Verified |
| Midpoint-vs-trapezoidal quadrature bias is a constant spatial pattern, not a temporal drift | Quadrature coefficients do not depend on time; the error structure is fixed by the grid and latitude | Verified (for a stationary flow; if U varies in time the error also varies, but its spatial mean is zero for a zonal-mean-preserving flow) |

## Best guess for the drift source (hypothesis, NOT a conclusion)

**Not a claim**. My best current reading, based on this review, is:

**The drift is NOT in our spectral preprocessing code. It is in the ERA5 LNSP field itself, expressed through the nonlinear `exp` transform.**

Reasoning (from probes cited in FromClaude.md, not independently verified by me):
1. `Σ dm / Σ m` over 24h matches the runtime tracer drift exactly, so the drift lives in the binary's `dm` field.
2. `dm` is a linear function of `Δps` because `m = (dA + dB*ps)*area/g` and our dA, dB are constant.
3. `Δps = Δexp(LNSP)` where LNSP is our T=359 band-limited gridpoint LNSP.
4. `<exp(LNSP)>` is not stationary because `LNSP` has time-varying spatial variance (and `exp` is convex).
5. The Poisson balance step only modifies am/bm, not m/dm; and our compute_mass_fluxes does not touch m at all. The only path from LNSP to m is `ps = exp(LNSP_on_grid)` and `m = (dA + dB*ps)*area/g`.
6. Therefore, the drift magnitude directly reflects `Δ<ps>` over 24h. A drift of 8.3e-6 corresponds to a global mean ps change of ~0.83 Pa.

**To verify or refute this hypothesis, the following diagnostics would settle it**:

(a) **Compute `<ps>` at each of the 24 windows of the v4 binary** (simple: `sum(ps*area)/sum(area)`) and compute `<ps>[24] - <ps>[1]`. Convert to mass via `Δmass = Δ<ps> * total_area / g`. If this matches `Σ Σdm = -4.26e13 kg`, the drift IS in ps, NOT in the spectral preprocessor.

(b) Compute `<LNSP_c00>` = the real part of the `(n=0, m=0)` spectral coefficient of LNSP at each window. This is the global mean of LNSP (not of ps). If this is constant across 24 hours but `<ps>` still drifts, the drift comes from changing spatial variance of LNSP (Jensen gap grows/shrinks). If `<LNSP_c00>` itself changes, the drift comes from a shift in the input ERA5 LNSP field.

(c) Run the same preprocessing with `LNSP_c00` forced constant across all 24 windows (e.g., subtract `(c00[t] - c00[0])` from each window's LNSP_c00). Observe whether the drift changes.

(d) Compare `<ps>` from our preprocessor to `<ps>` from an independent source (e.g., ECMWF IFS model output or MARS-retrieved `sp` GRIB field, if available). If our `<ps>` matches, the drift is in ERA5. If it doesn't, there's a bug in our `LNSP → ps` transform.

## Items flagged as "NOT TM5-faithful but probably not the drift source"

These are real divergences between our preprocessor and TM5 cy3-4dvar that the codebase has NOT previously documented as divergences:

1. **Midpoint-rule cell-face quadrature for mfu/mfv** (H1) — vs TM5's trapezoidal on a fine grid.
2. **`dp` from cell-center `exp(LNSP)` averaged at faces** (H1, M4) — vs TM5 evaluating `exp(LNSP)` spectrally at face locations on a fine grid.
3. **BFFT sampling at cell faces, labelled as cell centers** (H2) — half-cell spatial shift.
4. **No pole-row special case in `compute_mass_fluxes!`** (H5) — vs TM5's row-averaging at `lat ±π/2`.
5. **Extra VO(m,T) term included at (m, T-1)** (H4) — technically ours is more correct, but differs from TM5.
6. **`balance_mass_fluxes!` has no TM5 analog** (H6) — TM5 uses m-from-ps re-derivation, not Poisson forcing of am/bm.
7. **`merge_cell_field!` sums native am/bm then recomputes cm from merged fluxes** (M1) — TM5 sums am/bm via `FillLevels(combine,sum)` and then picks native cm via `FillLevels(combine,bottom)`. Mathematically equivalent but different code path.

The CLAUDE.md file previously claimed in several places (invariant 11, documentation comments) that the v4 path was "TM5-faithful". This is not accurate at the level of algorithmic detail. It is TM5-INSPIRED and achieves the same high-level goal (mass-conserving advection from spectral ERA5), but the quadrature, flux location, pole handling, balance scheme, and level merging all differ in specific ways. CLAUDE.md has since been updated in commit `759327a` with Rule 1c and a narrower framing for invariant 11.

## Verdict

**NEEDS INVESTIGATION — drift source NOT in the spectral preprocessor by my reading, but I cannot prove this without running the diagnostics (a)-(d) above.**

Concrete recommendations:
1. **Priority 1**: run diagnostic (a) — compute per-window `<ps>` directly from the v4 binary and confirm whether `Δ<ps> * total_area / g` matches the observed `ΣΣdm = -4.26e13 kg`. This is a 20-line script and will either confirm or refute the "drift is in the ERA5 LNSP" hypothesis definitively.
2. **Priority 2**: update CLAUDE.md invariant 11 if the drift turns out to be data-driven, to confirm the existing "source unknown" framing.
3. **Priority 3** (medium): fix H2 (the half-cell longitudinal shift). Either relabel `lons` to be `(i-1)*dlon_deg` (cell faces) or shift the spectral coefficient by `exp(i*m*dlon/2)` before BFFT. This will make local tracer fields comparable against TM5 runs on the same ERA5 input.
4. **Priority 3** (medium): add a pole-row special case to `compute_mass_fluxes!` (H5) — either row-averaging (TM5 style) or explicit zeroing of `am[:, 1, :], am[:, Ny, :]` in the preprocessor so the Poisson balance is not fed garbage.
5. **Priority 4** (low): at `n=T-1` in `vod2uv!`, document that we include a term TM5 omits. No correctness impact, but the difference is nonzero.

None of these is a "critical" bug in the drift sense. The drift is either (very likely) a property of the input ERA5 LNSP data, or (less likely) is caused by an interaction not yet understood. The spectral preprocessor's forward path from `LNSP → ps → m → dm` is a straightforward linear + exponential transform and no bug was found in it.

## Files NOT fully read (noted for transparency)

- `deps/tm5-cy3-4dvar/base/src/tmm_mf_ecmwf_tm5.F90` — read the spectral-field part (line 1013-1119), skipped the file header and non-spectral paths.
- `deps/tm5-cy3-4dvar/base/src/grid_type_sh.F90` — read `sh_vod2uv`, `shi_vod2uv`, `sh_Pnm`, `shgrid_Eval_Lons_*`; did not read `FourrierCoeff_*` routines.
- `deps/tm5-cy3-4dvar/base/src/grid_interpol.F90` — read `IntLat_sh_ll`, `IntLon_sh_ll`, `IntArea_sh_ll_fgh`, `ShRefinement`. Did not read `IntArea_shi_ll_fgh` (identical algorithm per scan). Did not read `IntArea_sh_ll_fh` or its `shi_ll_fh` variant.
- `deps/tm5-cy3-4dvar/base/src/advect_tools.F90` — read `dynam0` in full and `m2phlb1` start; only scanned `m2phlb`.
- `deps/tm5-cy3-4dvar/base/src/grid_type_hyb.F90` — read `levi_FillLevels_3d` in full; did not read `Compare`, `Init`, or level-info setup routines.

None of the unread portions contain spectral preprocessing algorithms. Would prioritize completing the read if there were a follow-up hypothesis specifically pointing to any of them.

--- End of report ---
