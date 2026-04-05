---
name: TM5 TimeInterpolation complete trace
description: Full trace of TM5 field interpolation architecture — what gets interpolated, how, and the dynam0 B-correction formula. Critical for ERA5 LL v4 binary NaN debugging.
type: reference
---

## TM5 Field Interpolation Architecture

### What TM5 interpolates

| Field | tinterp mode | Source |
|-------|-------------|--------|
| pu/pv (balanced horizontal mass flux, kg/s) | `interp3` (3-hourly linear) | meteo.F90:356-366, pycasso log:380 |
| sp2 (surface pressure, Pa) | `interp*` (must begin with 'interp') | meteo.F90:227-231 |
| spm (midpoint surface pressure) | computed as `alfa1*sp1 + alfa2*sp2` | meteo.F90:1842-1852 |
| mfu/mfv (raw mass fluxes, kg/s) | `interp3` | pycasso log:380 via 'ml' key |
| temperature | via `interp3` (ml) | pycasso log:380 |
| convection fields | `aver3` (3-hourly average) | pycasso log:381 |
| surface fields | `aver1` (1-hourly average) | pycasso log:379 |
| orography | `const` | pycasso log:382 |

### InterpolFractions formula (go_date.F90:2920-2971)
```
alfa2 = (t - t1) / (t2 - t1)
alfa1 = 1 - alfa2
```
Standard linear interpolation weights.

### TimeInterpolation for interp* (meteodata.F90:680-714)
```
tmid = tr(1) + (tr(2) - tr(1)) / 2     -- midpoint of substep
call InterpolFractions(tmid, tr1(1), tr2(1), alfa1, alfa2)
md%data = alfa1 * md%data1 + alfa2 * md%data2
```

### Check_CFL outer loop (advectm_cfl.F90:154-302)
```
do global_iteration
    do i = 1, n    -- n = number of substeps
        tr(1) = t1 + (i-1)*ndyn
        tr(2) = t1 + i*ndyn
        call Setup_MassFlow(tr, ndyn, status)   -- interpolates pu/pv, then dynam0
        call determine_cfl_iter(...)             -- checks CFL
    end do
    if .not. cfl_ok: ndyn = smaller value, rescale am_t/bm_t/cm_t
end do
```

### Setup_MassFlow (advectm_cfl.F90:313-380)
1. TimeInterpolation(pu_dat, tr)  -- interpolates pu to substep midpoint
2. TimeInterpolation(pv_dat, tr)  -- interpolates pv to substep midpoint
3. dynam0(n, ndyn)                -- computes am/bm/cm from interpolated pu/pv

### dynam0 formula (advect_tools.F90:638-809)
```
dtu = ndyn / (2 * tref)    -- half-step duration in seconds

conv_adv(i,j,l) = pu(i-1,j,l) - pu(i,j,l) + pv(i,j,l) - pv(i,j+1,l)  -- convergence (kg/s)
pit(i,j) = sum_l conv_adv(i,j,l)                                         -- column integral

-- B-correction from BOTTOM up:
sd(lmr-1) = conv_adv(lmr) - (bt(lmr) - bt(lmr+1)) * pit
sd(l)      = sd(l+1) + conv_adv(l+1) - (bt(l+1) - bt(l+2)) * pit

am = dtu * pu
bm = dtv * pv
cm = -dtw * sd   (only interior interfaces l=1..lmr-1)
```

### Key invariants
- bt(l) = hybrid B at interface l. bt(1)=1 (surface), bt(lm+1)=0 (TOA).
- cm(0) = 0 (surface boundary), cm(lm) = 0 (TOA boundary) -- by initial zeroing
- B-correction ensures sum_l(bt(l)-bt(l+1)) = 1, so column integral of sd cancels pit
- dtu = dtv = dtw (same timestep for all directions)
- pu/pv are BALANCED (mass-conserving) after BalanceMassFluxes in meteo.F90:1608

### Our v4 approach comparison
Our preprocessor stores am=half_dt*pu, bm=half_dt*pv (mass amounts, not fluxes).
Deltas: dam = am_next - am_curr, dm = m_next - m_curr.
Runtime: interpolates am/bm/m directly, recomputes cm from interpolated am/bm.
Key difference: TM5 interpolates raw fluxes then converts; we interpolate converted amounts.
Mathematically equivalent when half_dt is constant across all hours (which it is in our preprocessor).
