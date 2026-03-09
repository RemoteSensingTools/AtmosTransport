#!/usr/bin/env python3
"""
Download ERA5 convective mass flux on model levels for Tiedtke convection.

Downloads updraft (UDMF, param 71.162) and downdraft (DDMF, param 72.162) mass
flux on all 137 model levels, 6-hourly (00/06/12/18 UTC) at 0.5 deg resolution.

These are retrieved from `reanalysis-era5-complete` (MARS syntax). The output
is GRIB format (model-level diagnostic fields are only available as GRIB).
A GRIB-to-NetCDF conversion step runs automatically using cfgrib+xarray.

Output: one NetCDF per day in the specified output directory:
    era5_cmfmc_YYYYMMDD.nc

Usage:
    python3 scripts/downloads/download_era5_cmfmc.py \
        --start 2021-12-01 --end 2021-12-07 \
        --outdir ~/data/metDrivers/era5/cmfmc_catrine

Requirements:
    pip install cdsapi cfgrib eccodes xarray
    (and ~/.cdsapirc configured for CDS access)
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Download ERA5 convective mass flux on model levels")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--resolution", type=float, default=0.5,
                   help="Grid spacing in degrees (default: 0.5)")
    p.add_argument("--keep-grib", action="store_true",
                   help="Keep GRIB files after conversion")
    return p.parse_args()


def date_range(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def grib_to_netcdf(grib_path, nc_path):
    """Convert GRIB to NetCDF using cfgrib+xarray."""
    import xarray as xr
    # Open all variables; cfgrib may split into multiple datasets
    datasets = xr.open_datasets(grib_path, engine="cfgrib")
    # Merge and write
    ds = xr.merge(datasets)
    ds.to_netcdf(nc_path)
    ds.close()
    for d in datasets:
        d.close()
    print(f"    Converted to NetCDF: {os.path.basename(nc_path)}")


def download_cmfmc(client, date_str, resolution, outdir, keep_grib):
    """Download 6-hourly updraft+downdraft mass flux on 137 model levels."""
    nc_path = os.path.join(
        outdir, f"era5_cmfmc_{date_str.replace('-', '')}.nc")
    grib_path = nc_path.replace(".nc", ".grib")

    if os.path.exists(nc_path):
        sz = os.path.getsize(nc_path) / 1e6
        print(f"  SKIP (exists, {sz:.1f} MB): {os.path.basename(nc_path)}")
        return nc_path

    if not os.path.exists(grib_path):
        print(f"  Downloading UDMF+DDMF for {date_str}...")
        # MARS-style request for model-level diagnostic fields
        client.retrieve(
            "reanalysis-era5-complete",
            {
                "class": "ea",
                "expver": "1",
                "stream": "oper",
                "type": "an",
                "levtype": "ml",
                "levelist": "/".join(str(i) for i in range(1, 138)),
                "param": "71.162/72.162",
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
    end = datetime.strptime(args.end, "%Y-%m-%d")
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
    print("ERA5 Convective Mass Flux Downloader (Model Levels)")
    print(f"  Period:     {args.start} to {args.end}")
    print(f"  Resolution: {args.resolution} deg")
    print(f"  Output:     {outdir}")
    print(f"  Params:     71 (UDMF) + 72 (DDMF), 137 model levels, 6-hourly")
    print()

    n_days = (end - start).days + 1
    for i, dt in enumerate(date_range(start, end)):
        date_str = dt.strftime("%Y-%m-%d")
        print(f"[{i+1}/{n_days}] {date_str}")
        download_cmfmc(client, date_str, args.resolution, outdir,
                       args.keep_grib)

    print(f"\nDone! Downloaded {n_days} days to {outdir}")
    print("Next: merge into preprocessed file:")
    print(f"  julia --project=. scripts/preprocessing/merge_era5_cmfmc_to_massflux.jl \\")
    print(f"      <massflux.nc> {outdir}")


if __name__ == "__main__":
    main()
