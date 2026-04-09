#!/usr/bin/env python3
"""
Download ERA5 convective mass flux from MARS via CDS 'reanalysis-era5-complete'.

Uses ECMWF MARS access (requires CDS credentials with MARS access enabled).
Downloads parameterized convective updraft mass flux on model levels:
  param 78.162 = convective updraft mass flux [kg/(m²·s)]

These are forecast fields available at 3-hourly steps from 06/18 UTC.
We retrieve steps 3/6/9/12 from 06/18 UTC base times to cover all 6-hourly
met windows (00/06/12/18 UTC).

Step → valid time:
  06 UTC + 3h → 09 UTC  (covers 06–12 window midpoint)
  06 UTC + 6h → 12 UTC  (12 UTC window)
  18 UTC + 3h → 21 UTC  (covers 18–00 window midpoint)
  18 UTC + 6h → 00 UTC  (00 UTC next day window)

For the postprocessor, valid times 00/06/12/18 UTC are linearly interpolated
from the two bracketing 3-hourly values.

Output: one GRIB file per day in the output directory.

Usage:
    python3 scripts/download_era5_mars_cmfmc.py \\
        --start 2023-06-01 --end 2023-06-30 \\
        --outdir ~/data/metDrivers/era5/cmfmc_june2023

Requires: pip install cdsapi  (and ~/.cdsapirc configured with MARS access)
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Download ERA5 convective mass flux from MARS")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end",   required=True, help="End date YYYY-MM-DD")
    p.add_argument("--outdir", required=True, help="Output directory for GRIB files")
    p.add_argument("--resolution", type=float, default=0.5,
                   help="Grid spacing in degrees (default: 0.5)")
    p.add_argument("--levelist", default="1/to/137",
                   help="Model levels, e.g. '1/to/137' (default all 137)")
    return p.parse_args()


def date_range(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def download_cmfmc_day(client, date_str, resolution, levelist, outdir):
    """Download convective mass flux for one day.

    ERA5 CMFMC uses param 77.128 (shortName 'mflx' = moist convective mass flux,
    TM5 convention, [kg m-2 s-1]).  This is in the forecast stream:
    base times 06 and 18 UTC, steps 3/6/9/12.
    Valid times: 09, 12, 21, 00 (next day).
    """
    outfile = os.path.join(outdir, f"era5_cmfmc_{date_str.replace('-', '')}.grib")
    if os.path.exists(outfile) and os.path.getsize(outfile) > 1024:
        sz = os.path.getsize(outfile) / 1e6
        print(f"  SKIP cmfmc (exists, {sz:.0f} MB): {os.path.basename(outfile)}")
        return outfile

    print(f"  Downloading CMFMC for {date_str}...")

    # Build level list string
    if levelist == "1/to/137":
        levelist_str = "/".join(str(l) for l in range(1, 138))
    else:
        levelist_str = levelist

    client.retrieve(
        "reanalysis-era5-complete",
        {
            "class":       "ea",
            "date":        date_str,
            "expver":      "1",
            "levelist":    levelist_str,
            "levtype":     "ml",
            "param":       "77.128",       # mflx = moist convective mass flux [kg m-2 s-1]
            "stream":      "oper",
            "time":        "06:00:00/18:00:00",
            "step":        "3/6/9/12",     # → valid at 09/12/21/00+1day
            "type":        "fc",
            "grid":        f"{resolution}/{resolution}",
            "data_format": "grib",
        },
        outfile,
    )
    sz = os.path.getsize(outfile) / 1e6
    print(f"  ✓ {os.path.basename(outfile)} ({sz:.0f} MB)")
    return outfile


def main():
    args = parse_args()
    start  = datetime.strptime(args.start, "%Y-%m-%d")
    end    = datetime.strptime(args.end,   "%Y-%m-%d")
    outdir = os.path.expanduser(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    try:
        import cdsapi
    except ImportError:
        print("ERROR: cdsapi not installed. Run: pip install cdsapi")
        sys.exit(1)

    rc = Path.home() / ".cdsapirc"
    if not rc.exists():
        print("ERROR: ~/.cdsapirc not found. Configure CDS API credentials.")
        sys.exit(1)

    client = cdsapi.Client()
    print(f"ERA5 Convective Mass Flux (MARS) Downloader")
    print(f"  Period:     {args.start} to {args.end}")
    print(f"  Resolution: {args.resolution}°")
    print(f"  Levels:     {args.levelist}")
    print(f"  Output:     {outdir}")
    print()

    n_days = (end - start).days + 1
    for i, dt in enumerate(date_range(start, end)):
        date_str = dt.strftime("%Y-%m-%d")
        print(f"[{i+1}/{n_days}] {date_str}")
        download_cmfmc_day(client, date_str, args.resolution, args.levelist, outdir)

    print(f"\nDone! Downloaded {n_days} days to {outdir}")


if __name__ == "__main__":
    main()
