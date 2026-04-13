#!/usr/bin/env python3
"""
Download ERA5 specific humidity (QV) on model levels via CDS API.

QV (param 133) is a gridpoint field, not spectral. Needed for dry-mass
computation: m_dry = DELP * (1 - QV) * area / g.

Downloads 6-hourly QV at 0.5 degree resolution on all 137 model levels.
Output: one NetCDF per month.

Note: download_era5_physics.py --fields thermodynamics also downloads QV
(hourly, along with temperature) as part of the unified physics pipeline.
This standalone script remains for backward compatibility.

Usage:
    python3 scripts/downloads/download_era5_qv.py --start 2021-12-01 --end 2021-12-31
"""

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Download ERA5 QV on model levels")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--outdir", default=os.path.expanduser("~/data/AtmosTransport/met/era5/qv"),
                   help="Output directory")
    p.add_argument("--resolution", default="0.5/0.5", help="Grid resolution")
    return p.parse_args()

def download_month(year, month, days, outdir, resolution):
    """Download one month of QV data."""
    import cdsapi
    client = cdsapi.Client()

    outfile = os.path.join(outdir, f"era5_qv_{year}{month:02d}.nc")
    if os.path.exists(outfile):
        size_mb = os.path.getsize(outfile) / 1e6
        print(f"  Already exists: {outfile} ({size_mb:.0f} MB)")
        return

    day_list = [f"{d:02d}" for d in days]
    print(f"  Downloading {year}-{month:02d} days {day_list[0]}-{day_list[-1]} ...")

    client.retrieve("reanalysis-era5-complete", {
        "class": "ea",
        "type": "an",
        "stream": "oper",
        "expver": "1",
        "levtype": "ml",
        "levelist": "/".join(str(i) for i in range(1, 138)),
        "param": "133",  # QV = specific humidity
        "date": f"{year}-{month:02d}-{day_list[0]}/to/{year}-{month:02d}-{day_list[-1]}",
        "time": "00:00:00/06:00:00/12:00:00/18:00:00",
        "grid": resolution,
    }, outfile)

    size_mb = os.path.getsize(outfile) / 1e6
    print(f"  Done: {outfile} ({size_mb:.0f} MB)")

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    print(f"ERA5 QV download: {args.start} to {args.end}")
    print(f"  Output: {args.outdir}")

    # Group by month
    from collections import defaultdict
    months = defaultdict(list)
    d = start
    while d <= end:
        months[(d.year, d.month)].append(d.day)
        d += timedelta(days=1)

    for (year, month), days in sorted(months.items()):
        download_month(year, month, days, args.outdir, args.resolution)

    print("All downloads complete.")

if __name__ == "__main__":
    main()
