#!/usr/bin/env python3
"""Plan 45 Stage-4 verdict: did global-mean referencing remove the co2_natural
F32 surplus?

Compares the A/B pair (raw vs referenced LinRood December campaign):
- co2_natural: dm(t) vs cumulative lmdz |F(t); end-of-month surplus as % of
  emission (PPM-path baseline was +28%).
- sf6: emission deficit both runs (constant EDGAR rate).
- co2_fossil: deficit both runs (IC=0 control; referencing must not regress it).

Usage: stage4_surplus_verdict.py raw.nc ref.nc
"""
import os
import sys
import numpy as np
import netCDF4 as nc

M_AIR = 28.9644
M_CO2 = 44.01
M_SF6 = 146.055
KGC_TO_KGCO2 = 44.01 / 12.011
SF6_RATE = 3.239166e-1          # kg/s (EDGAR * 1.0116635 scale)
FOSSIL_RATE = 1.229399e6        # kg/s (gridfed Dec 2021)
LMDZ = os.path.expanduser(
    "~/data/AtmosTransport/catrine/Emissions/LMDZ_fluxes/"
    "z_cams_l_cams55_202112_FT24r2_ra_sfc_3h_co2_flux.nc")

raw_p, ref_p = sys.argv[1], sys.argv[2]

with nc.Dataset(LMDZ) as fz:
    flux = np.asarray(fz.variables["flux_apos"][:], dtype=np.float64)
    area = np.asarray(fz.variables["area"][:], dtype=np.float64)
nat_rate = np.array([np.sum(flux[t] * area) for t in range(flux.shape[0])]) * KGC_TO_KGCO2
cum_nat = np.concatenate([[0.0], np.cumsum(nat_rate * 10800.0)])

def series(path, tracer, M):
    f = nc.Dataset(path)
    t = np.array(f.variables["time"][:])
    n = len(t)
    am = f.variables["air_mass"]
    q = f.variables[tracer]
    Ms = np.empty(n)
    for i in range(n):
        Ms[i] = np.sum(np.asarray(q[i], dtype=np.float64) *
                       np.asarray(am[i], dtype=np.float64)) * M / M_AIR
    nan = bool(np.isnan(np.asarray(q[-1])).any())
    f.close()
    return (t - t[0]), Ms, nan

print("=== co2_natural (time-varying lmdz flux) ===")
for lbl, path in (("raw", raw_p), ("ref", ref_p)):
    th, Ms, nan = series(path, "co2_natural", M_CO2)
    dm = Ms - Ms[0]
    F = cum_nat[:len(th)]
    print(f"  {lbl}: NaN={nan}")
    for i in (8, 56, 112, len(th) - 1):       # 24h, 1wk, 2wk, end
        if i < len(th) and F[i] > 0:
            s = dm[i] - F[i]
            print(f"    t={th[i]:5.0f}h  dm={dm[i]/1e12:8.4f}Pg  |F={F[i]/1e12:7.4f}Pg  "
                  f"surplus={s/1e12:+8.4f}Pg ({s/F[i]*100:+7.2f}% of emission)")

print("=== sf6 / co2_fossil (constant rates) ===")
for tracer, M, rate in (("sf6", M_SF6, SF6_RATE), ("co2_fossil", M_CO2, FOSSIL_RATE)):
    line = f"  {tracer:11s}:"
    for lbl, path in (("raw", raw_p), ("ref", ref_p)):
        th, Ms, nan = series(path, tracer, M)
        dur = th[-1] * 3600.0
        d = 1.0 - (Ms[-1] - Ms[0]) / (rate * dur)
        line += f"  {lbl} deficit {d*100:+.3f}%{' NaN!' if nan else ''}"
    print(line)
print("done")
