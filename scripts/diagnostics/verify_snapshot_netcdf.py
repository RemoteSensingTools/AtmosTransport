#!/usr/bin/env python3
"""
Rigorous raw-value verification for src_v2 snapshot NetCDF files.

Prints min/max/mean, NaN count, Inf count, and out-of-range cell count for
each timestep in `co2_surface` and `co2_column_mean`. Indexing follows the
actual netCDF dimension order — NOT `var[i]` which was the 2026-04-09 bug.

Usage:
    python3 scripts/diagnostics/verify_snapshot_netcdf.py <file.nc> \
        [vmin vmax] [--mass-tol=1e-9]

Defaults to Catrine CO2 physical range [3.9e-4, 4.5e-4] and a F64 mass
drift threshold of 1e-9. For F32 runs pass `--mass-tol=1e-6` (F32 noise
floor) to avoid false-positive failures on single-precision runs.
"""
import sys
import numpy as np
import netCDF4 as nc


def slice_time(var, i, grid_type):
    dims = var.dimensions
    # latlon layouts seen in export_transport_v2_snapshots.jl
    if dims == ("time", "lat", "lon"):
        return np.asarray(var[i, :, :])
    if dims == ("lat", "lon", "time"):
        return np.asarray(var[:, :, i])
    if dims == ("lon", "lat", "time"):
        return np.asarray(var[:, :, i]).T
    if dims == ("time", "lon", "lat"):
        return np.asarray(var[i, :, :]).T
    # reduced gaussian
    if dims == ("time", "cell"):
        return np.asarray(var[i, :])
    if dims == ("cell", "time"):
        return np.asarray(var[:, i])
    raise ValueError(f"Unsupported dims {dims!r} for grid {grid_type!r}")


def summary(name, arr, vmin, vmax):
    n = arr.size
    nan = int(np.isnan(arr).sum())
    inf = int(np.isinf(arr).sum())
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return f"{name}: ALL NON-FINITE (nan={nan}, inf={inf}, n={n})"
    lo = float(finite.min())
    hi = float(finite.max())
    mean = float(finite.mean())
    below = int((finite < vmin).sum())
    above = int((finite > vmax).sum())
    flag = " OK" if (nan == 0 and inf == 0 and below == 0 and above == 0) else " FAIL"
    return (
        f"{name}:{flag} min={lo:.3e} max={hi:.3e} mean={mean:.3e} "
        f"nan={nan} inf={inf} <{vmin:.2e}={below} >{vmax:.2e}={above} (n={n})"
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: verify_snapshot_netcdf.py <file.nc> [vmin vmax] [--mass-tol=1e-9]")
        sys.exit(1)
    path = sys.argv[1]
    positional = [a for a in sys.argv[2:] if not a.startswith("--")]
    vmin = float(positional[0]) if len(positional) > 0 else 3.9e-4
    vmax = float(positional[1]) if len(positional) > 1 else 4.5e-4
    mass_tol = 1e-9
    for arg in sys.argv[2:]:
        if arg.startswith("--mass-tol="):
            mass_tol = float(arg.split("=", 1)[1])

    ds = nc.Dataset(path, "r")
    try:
        grid_type = getattr(ds, "grid_type", "unknown")
        scheme = getattr(ds, "scheme", "unknown")
        tracer = getattr(ds, "tracer_name", "unknown")
        times = np.asarray(ds.variables["time_hours"][:])
        sfc_var = ds.variables["co2_surface"]
        col_var = ds.variables["co2_column_mean"]

        print(f"File:     {path}")
        print(f"Grid:     {grid_type}  scheme={scheme}  tracer={tracer}")
        print(f"Range OK: [{vmin:.2e}, {vmax:.2e}]")
        print(f"Times:    {list(times)}")
        print(f"sfc dims: {sfc_var.dimensions}  shape={sfc_var.shape}")
        print(f"col dims: {col_var.dimensions}  shape={col_var.shape}")

        any_fail = False
        for i, t in enumerate(times):
            sfc = slice_time(sfc_var, i, grid_type)
            col = slice_time(col_var, i, grid_type)
            s1 = summary("surface", sfc, vmin, vmax)
            s2 = summary("column ", col, vmin, vmax)
            print(f"\nt = {float(t):6.2f} h")
            print(f"  {s1}")
            print(f"  {s2}")
            if "FAIL" in s1 or "FAIL" in s2:
                any_fail = True

        if "air_mass_total" in ds.variables and "tracer_mass_total" in ds.variables:
            am = np.asarray(ds.variables["air_mass_total"][:])
            tm = np.asarray(ds.variables["tracer_mass_total"][:])
            am0 = am[0] if am[0] != 0 else 1.0
            tm0 = tm[0] if tm[0] != 0 else 1.0
            am_drift = (am - am[0]) / am0
            tm_drift = (tm - tm[0]) / tm0
            print("\nGlobal mass (relative to t=0):")
            for i, t in enumerate(times):
                print(
                    f"  t={float(t):6.2f}h  air={am[i]:.6e} ({am_drift[i]:+.2e})  "
                    f"tracer={tm[i]:.6e} ({tm_drift[i]:+.2e})"
                )
            if np.any(np.abs(am_drift) > mass_tol):
                print(f"  WARNING: air mass drift > {mass_tol:.1e} relative")
                any_fail = True
            if np.any(np.abs(tm_drift) > mass_tol):
                print(f"  WARNING: tracer mass drift > {mass_tol:.1e} relative")
                any_fail = True

        if any_fail:
            print("\nVERDICT: FAIL (see above)")
            sys.exit(2)
        else:
            print("\nVERDICT: PASS")
    finally:
        ds.close()


if __name__ == "__main__":
    main()
