#!/usr/bin/env python3
"""Plan 45 Stage-2/3 smoke analysis: referenced (anomaly-transport) run vs the
raw run of the same config. Expectations: NO NaN; tracer field agrees to F32
reconstruction noise; the emission budget closes at least as well.

Usage: compare_referenced_vs_raw.py raw.nc referenced.nc tracer rate_kg_s [M_tracer]
"""
import sys
import numpy as np
import netCDF4 as nc

M_AIR = 28.9644
raw_p, ref_p, tracer = sys.argv[1], sys.argv[2], sys.argv[3]
rate = float(sys.argv[4])
M_tr = float(sys.argv[5]) if len(sys.argv) > 5 else 146.055

fr, ff = nc.Dataset(raw_p), nc.Dataset(ref_p)
t = np.array(fr.variables["time"][:])
dur = (t[-1] - t[0]) * 3600.0

ok = True
for lbl, f in (("raw", fr), ("referenced", ff)):
    q = np.asarray(f.variables[tracer][-1])
    if np.isnan(q).any():
        print(f"{lbl}: NaN in final {tracer} field!")
        ok = False

# field agreement at final frame (VMR)
qr = np.asarray(fr.variables[tracer][-1], dtype=np.float64)
qf = np.asarray(ff.variables[tracer][-1], dtype=np.float64)
scale = np.abs(qr).max()
dmax = np.abs(qf - qr).max()
print(f"final-frame {tracer}: max|Δvmr| = {dmax:.3e}  (max|vmr| = {scale:.3e}, "
      f"rel {dmax/scale:.3e})")

# budget change dm (kg) for both runs, directly — no emission normalization,
# so the metric works for no-flux runs (rate = 0) too.
def dmass(f):
    am0 = np.asarray(f.variables["air_mass"][0], dtype=np.float64)
    amL = np.asarray(f.variables["air_mass"][-1], dtype=np.float64)
    q0 = np.asarray(f.variables[tracer][0], dtype=np.float64)
    qL = np.asarray(f.variables[tracer][-1], dtype=np.float64)
    dm = (np.sum(qL * amL) - np.sum(q0 * am0)) * M_tr / M_AIR
    burden0 = np.sum(q0 * am0) * M_tr / M_AIR
    return dm, burden0

dm_raw, burden0 = dmass(fr)
dm_ref, _ = dmass(ff)
if rate > 0:
    print(f"emission deficit: raw {(1 - dm_raw/(rate*dur))*100:+.4f}%   "
          f"referenced {(1 - dm_ref/(rate*dur))*100:+.4f}%")

# Acceptance is on ABSOLUTE drift relative to the burden, not on the deficit
# delta in %-of-emission: for a small-emission tracer (sf6: emission ~1e-4 of
# burden/day) the percentage magnifies kg-scale F32 noise that is irrelevant
# at the conservation scale referencing targets. Bisection (plan 45 stage 2/3)
# measured the referenced overhead at ~3e-7 of burden per day — gate at 2e-6.
extra = abs(dm_ref - dm_raw)                     # kg of extra non-closure
days = dur / 86400.0
print(f"referenced-vs-raw budget difference: {extra:.1f} kg "
      f"({extra/burden0/max(days,1e-9):.2e} of burden per day)")
if extra / burden0 > 2e-6 * days:
    print("FAIL: referenced run's extra drift exceeds the F32 noise-floor gate")
    ok = False

fr.close(); ff.close()
print("SMOKE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
