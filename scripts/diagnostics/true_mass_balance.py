#!/usr/bin/env python3
"""TRUE mass balance for a constant-rate tracer: compare the FULL physical
tracer-mass change to the emission the MODEL ACTUALLY APPLIED (its logged
model-storage rate), in storage units — no hardcoded inventory rate, no
molar-mass assumption (the M_tracer/M_AIR factor cancels).

Usage: true_mass_balance.py <out.nc> <run.log> <tracer>
"""
import sys, re, numpy as np, netCDF4 as nc
ncf, logf, tracer = sys.argv[1], sys.argv[2], sys.argv[3]
# the model's actual applied rate (storage units, kg_air_equiv/s)
rate = None
pat = re.compile(rf"Surface source {re.escape(tracer)} total model-storage rate:\s*([0-9.eE+-]+)")
for line in open(logf, errors="ignore"):
    m = pat.search(line)
    if m: rate = float(m.group(1))
if rate is None:
    sys.exit(f"could not find logged model-storage rate for {tracer} in {logf}")
f = nc.Dataset(ncf); t = np.array(f.variables["time"][:]); dur = (t[-1]-t[0])*3600.0
am = f.variables["air_mass"]; q = f.variables[tracer]
# storage-unit total mass = Sum(vmr * air_mass) over the full written field (F64)
def stored(i):
    return np.sum(np.asarray(q[i],dtype=np.float64)*np.asarray(am[i],dtype=np.float64))
m0, mN = stored(0), stored(-1)
emitted = rate * dur                       # storage units (kg_air_equiv)
dm = mN - m0
print(f"{tracer}: applied rate = {rate:.9e} kg_air_equiv/s,  dur = {dur/86400:.2f} d")
print(f"  Δmass (stored)   = {dm:.6e}")
print(f"  emitted (model)  = {emitted:.6e}")
print(f"  TRUE imbalance   = {(dm-emitted):.4e}  =  {(1-dm/emitted)*100:+.4f}% of emission")
print(f"  NaN: {bool(np.isnan(np.asarray(q[-1])).any())}")
f.close()
