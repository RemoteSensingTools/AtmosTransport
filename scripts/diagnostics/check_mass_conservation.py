#!/usr/bin/env python3
"""Total tracer-mass conservation check for a no-flux (and ideally no-chemistry)
run: total mass = sum over all cells & levels of VMR * dry air_mass * M/M_air,
computed in float64 from the 3D fields and the dry layer air mass. With no
emission/chemistry the total of each tracer MUST be constant; the printed
max|drift| / mass(t0) is the conservation error (in F64, <~1e-12 = clean;
anything larger is an algorithmic transport leak).

  python3 scripts/diagnostics/check_mass_conservation.py <run.nc>
"""
import sys
import numpy as np
from netCDF4 import Dataset

NC = sys.argv[1]
M_AIR = 28.9644
TR = {"co2_natural": 44.01, "co2_fossil": 44.01, "sf6": 146.06, "rn222": 222.0}

d = Dataset(NC)
nt = d.variables["air_mass"].shape[0]
print(f"{NC}\n{nt} frames; total tracer mass conservation (no-flux pure transport):\n")
print(f"  {'tracer':12} {'mass(t0) [kg]':>17} {'mass(t_end) [kg]':>17} {'max|drift|/m0':>15}  verdict")
for v, M in TR.items():
    if v not in d.variables:
        continue
    mr = M / M_AIR
    m = np.array([
        float(np.sum(np.asarray(d.variables[v][t], dtype=np.float64)
                     * np.asarray(d.variables["air_mass"][t], dtype=np.float64))) * mr
        for t in range(nt)])
    drift = (m - m[0]) / m[0] if m[0] != 0 else m - m[0]
    mx = float(np.max(np.abs(drift)))
    verdict = "CONSERVED" if mx < 1e-12 else ("~F32 floor" if mx < 1e-5 else "LEAKS")
    print(f"  {v:12} {m[0]:17.9e} {m[-1]:17.9e} {mx:15.3e}  {verdict}")
    step = max(1, nt // 8)
    print(f"     drift @ frames 0..{nt-1}: " + " ".join(f"{x:+.1e}" for x in drift[::step]))
d.close()
