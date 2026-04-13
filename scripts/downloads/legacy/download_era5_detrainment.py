#!/usr/bin/env python3
"""
** SUPERSEDED by download_era5_physics.py (2026-03-31) **

This script used params 214.162/215.162 (UDRF/DDRF) which do NOT exist in ERA5
experiment version 1, and incorrectly assumed they were analysis fields.
ECMWF confirmed the correct parameters are forecast fields:
  - 235011: Time-mean updraught detrainment rate
  - 235012: Time-mean downdraught detrainment rate

Use instead:
    python3 scripts/downloads/download_era5_physics.py \\
        --start 2021-12-01 --end 2021-12-07 \\
        --outdir ~/data/AtmosTransport/met/era5/physics \\
        --fields convection

--- Original docstring (for reference) ---

Download ERA5 updraft/downdraft detrainment rates on model levels.
Used params 214.162/215.162 which do not exist in ERA5 expver 1.
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Download ERA5 convective detrainment rates on model levels")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--resolution", type=float, default=0.5,
                   help="Grid spacing in degrees (default: 0.5)")
    p.add_argument("--keep-grib", action="store_true",
                   help="Keep GRIB files after conversion")
    return p.parse_args()


def date_range(start, end_date):
    d = start
    while d <= end_date:
        yield d
        d += timedelta(days=1)


def grib_to_netcdf(grib_path, nc_path):
    """Convert GRIB to NetCDF using cfgrib+xarray."""
    import xarray as xr
    ds = xr.open_dataset(grib_path, engine="cfgrib")
    ds.to_netcdf(nc_path)
    ds.close()
    print(f"    Converted to NetCDF: {os.path.basename(nc_path)}")


def download_detrainment(client, date_str, resolution, outdir, keep_grib):
    """Download 6-hourly updraft/downdraft detrainment rates on 137 model levels."""
    nc_path = os.path.join(
        outdir, f"era5_detr_{date_str.replace('-', '')}.nc")
    grib_path = nc_path.replace(".nc", ".grib")

    if os.path.exists(nc_path):
        sz = os.path.getsize(nc_path) / 1e6
        print(f"  SKIP (exists, {sz:.1f} MB): {os.path.basename(nc_path)}")
        return nc_path

    if not os.path.exists(grib_path):
        print(f"  Downloading UDRF+DDRF for {date_str}...")
        # MARS-style request for model-level diagnostic fields
        # Param 214.162 = updraft detrainment rate [kg/m3/s]
        # Param 215.162 = downdraft detrainment rate [kg/m3/s]
        client.retrieve(
            "reanalysis-era5-complete",
            {
                "class": "ea",
                "expver": "1",
                "stream": "oper",
                "type": "an",
                "levtype": "ml",
                "levelist": "/".join(str(i) for i in range(1, 138)),
                "param": "214.162/215.162",
                "date": date_str,
                "time": "00:00:00/06:00:00/12:00:00/18:00:00",
                "grid": f"{resolution}/{resolution}",
            },
            grib_path,
        )
        sz = os.path.getsize(grib_path) / 1e6
        print(f"    GRIB: {os.path.basename(grib_path)} ({sz:.1f} MB)")

    # Convert GRIB to NetCDF
    grib_to_netcdf(grib_path, nc_path)
    if not keep_grib:
        os.remove(grib_path)

    return nc_path


def main():
    args = parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end_date = datetime.strptime(args.end, "%Y-%m-%d")
    outdir = os.path.expanduser(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    for pkg in ["cdsapi", "cfgrib", "xarray"]:
        try:
            __import__(pkg)
        except ImportError:
            print(f"ERROR: {pkg} not installed. Run: pip install cdsapi cfgrib eccodes xarray")
            sys.exit(1)

    import cdsapi

    rc = Path.home() / ".cdsapirc"
    if not rc.exists():
        print("ERROR: ~/.cdsapirc not found. Configure CDS API credentials.")
        sys.exit(1)

    client = cdsapi.Client()
    print("ERA5 Convective Detrainment Rate Downloader (Model Levels)")
    print(f"  Period:     {args.start} to {args.end}")
    print(f"  Resolution: {args.resolution} deg")
    print(f"  Output:     {outdir}")
    print(f"  Params:     214 (UDRF) + 215 (DDRF), 137 model levels, 6-hourly")
    print()

    n_days = (end_date - start).days + 1
    for i, dt in enumerate(date_range(start, end_date)):
        date_str = dt.strftime("%Y-%m-%d")
        print(f"[{i+1}/{n_days}] {date_str}")
        download_detrainment(client, date_str, args.resolution, outdir,
                             args.keep_grib)

    print(f"\nDone! Downloaded {n_days} days to {outdir}")
    print("Next: preprocess TM5 convection fields:")
    print(f"  julia --project=. scripts/preprocessing/preprocess_era5_tm5_convection.jl \\")
    print(f"      <massflux.nc> <cmfmc_dir> {outdir}")


if __name__ == "__main__":
    main()
