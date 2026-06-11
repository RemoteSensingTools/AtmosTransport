#!/usr/bin/env python3
"""Bit-identity gate for plan 45: compare every variable of two NetCDF outputs
byte-for-byte (data only — file-level hashes are confounded by creation
metadata). Exit 0 = identical, 1 = any mismatch.

Usage: compare_nc_bitident.py a.nc b.nc
"""
import sys
import numpy as np
import netCDF4 as nc

if len(sys.argv) != 3:
    sys.exit(f"usage: {sys.argv[0]} a.nc b.nc")
a_path, b_path = sys.argv[1], sys.argv[2]
fa, fb = nc.Dataset(a_path), nc.Dataset(b_path)

va, vb = set(fa.variables), set(fb.variables)
ok = True
if va != vb:
    print(f"variable sets differ: only-in-a={sorted(va-vb)} only-in-b={sorted(vb-va)}")
    ok = False

for name in sorted(va & vb):
    xa, xb = fa.variables[name], fb.variables[name]
    if xa.shape != xb.shape:
        print(f"{name}: shape {xa.shape} != {xb.shape}")
        ok = False
        continue
    da = np.asarray(xa[:])
    db = np.asarray(xb[:])
    if da.tobytes() == db.tobytes():
        print(f"{name}: IDENTICAL ({da.size} values)")
    else:
        diff = np.flatnonzero(da.ravel() != db.ravel())
        try:
            mx = np.nanmax(np.abs(da.astype(np.float64) - db.astype(np.float64)))
            detail = f"max|diff|={mx:.3e}"
        except (TypeError, ValueError):
            detail = "(non-numeric variable)"
        print(f"{name}: MISMATCH at {diff.size}/{da.size} values, {detail}")
        ok = False

fa.close(); fb.close()
print("GATE:", "PASS (bit-identical)" if ok else "FAIL")
sys.exit(0 if ok else 1)
