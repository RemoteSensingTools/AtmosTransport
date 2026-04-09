#!/usr/bin/env python3
"""
Download ERA5 hourly surface pressure (param=134) at 0.5 degrees for
cross-checking against our spectrally-synthesized ps in the v4 binary.

Uses the CDS single-levels dataset (NOT the MARS spectral stream), so
it's a plain gridpoint request with no spectral coefficients. The CDS
backend regrids from ERA5's native reduced Gaussian grid (N320, ~0.28
degree) to the requested regular lat/lon grid.

Output: single day of 24 hourly snapshots, ~50 MB/day.

Usage:
    python scripts/downloads/download_era5_sp.py [YYYY-MM-DD]

If no date is given, downloads 2021-12-01.
"""
import os
import sys
import cdsapi

OUTDIR = os.path.expanduser("~/data/AtmosTransport/met/era5/surface_pressure")
os.makedirs(OUTDIR, exist_ok=True)

if len(sys.argv) > 1:
    date = sys.argv[1]
else:
    date = "2021-12-01"

year, month, day = date.split("-")
datestr = year + month + day
outfile = os.path.join(OUTDIR, f"era5_sp_{datestr}.nc")

if os.path.exists(outfile) and os.path.getsize(outfile) > 1_000_000:
    print(f"SKIP (exists): {outfile} ({os.path.getsize(outfile)/1e6:.0f} MB)",
          flush=True)
    sys.exit(0)

print(f"Requesting ERA5 surface pressure for {date} at 0.5 deg...", flush=True)

c = cdsapi.Client()
c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "variable": "surface_pressure",
        "year": year,
        "month": month,
        "day": day,
        "time": [f"{h:02d}:00" for h in range(24)],
        "format": "netcdf",
        "grid": [0.5, 0.5],      # 0.5 deg regular lat-lon
        "area": [90, -180, -90, 180],  # global
    },
    outfile,
)

sz_mb = os.path.getsize(outfile) / 1e6
print(f"OK: {outfile} ({sz_mb:.0f} MB)", flush=True)
