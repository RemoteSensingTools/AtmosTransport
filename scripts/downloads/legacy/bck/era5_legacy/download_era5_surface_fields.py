#!/usr/bin/env python3
"""
Download ERA5 surface and convective fields for PBL diffusion + Tiedtke convection.

Fields:
  Surface (hourly analysis, 0.5° lat-lon):
    - BLH   (param 159) — boundary layer height [m]
    - SSHF  (param 146) — surface sensible heat flux [J/m²] (accumulated → deaccumulated)
    - T2M   (param 167) — 2m temperature [K]
    - U10   (param 165) — 10m U wind [m/s]
    - V10   (param 166) — 10m V wind [m/s]
    - ZUST  (param 3033) — friction velocity [m/s] (for PBL diffusion)

  Convective mass flux: ** SUPERSEDED by download_era5_physics.py **
    Old params 71.162/72.162 don't exist in ERA5.
    Use download_era5_physics.py --fields convection for correct params 235009-235012.

Output: one NetCDF file per day in the specified output directory.

Usage:
    python3 scripts/download_era5_surface_fields.py \\
        --start 2023-06-01 --end 2023-06-30 \\
        --outdir ~/data/metDrivers/era5/surface_fields_june2023

Requires: pip install cdsapi  (and ~/.cdsapirc configured)
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Download ERA5 surface + convective fields")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--resolution", type=float, default=0.5,
                   help="Grid spacing in degrees (default: 0.5)")
    return p.parse_args()


def date_range(start, end):
    """Generate dates from start to end (inclusive)."""
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def download_surface_fields(client, date_str, resolution, outdir):
    """Download hourly surface analysis fields for one day."""
    outfile = os.path.join(outdir, f"era5_surface_{date_str.replace('-', '')}.nc")
    if os.path.exists(outfile):
        sz = os.path.getsize(outfile) / 1e6
        print(f"  SKIP surface (exists, {sz:.1f} MB): {os.path.basename(outfile)}")
        return outfile

    print(f"  Downloading surface fields for {date_str}...")
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "boundary_layer_height",
                "surface_sensible_heat_flux",
                "2m_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "friction_velocity",
            ],
            "year": date_str[:4],
            "month": date_str[5:7],
            "day": date_str[8:10],
            "time": [f"{h:02d}:00" for h in range(0, 24)],
            "grid": [resolution, resolution],
            "format": "netcdf",
        },
        outfile,
    )
    sz = os.path.getsize(outfile) / 1e6
    print(f"  ✓ {os.path.basename(outfile)} ({sz:.1f} MB)")
    return outfile


def download_convective_fields(client, date_str, resolution, outdir):
    """Download 3-hourly updraft/downdraft mass flux for one day.

    ERA5 convective mass fluxes are available as instantaneous model-level
    fields (param 71.162 / 72.162) at pressure levels, or from the forecast
    stream on single levels. The cleanest source for column-integrated
    convective transport is the pressure-level product.

    For simplicity, we download the single-level proxies here. Users with
    MARS access should retrieve model-level UDMF/DDMF directly.
    """
    outfile = os.path.join(
        outdir, f"era5_convective_{date_str.replace('-', '')}.nc")
    if os.path.exists(outfile):
        sz = os.path.getsize(outfile) / 1e6
        print(f"  SKIP convective (exists, {sz:.1f} MB): "
              f"{os.path.basename(outfile)}")
        return outfile

    print(f"  Downloading convective mass flux for {date_str}...")

    # Convective fields are in the forecast stream, retrieved at
    # forecast steps 3/6/9/12 from base times 06:00 and 18:00 UTC.
    # This gives 8 timesteps per day (3-hourly).
    client.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "vertical_velocity",  # omega [Pa/s] — proxy for convective flux
            ],
            "pressure_level": [
                "1", "2", "3", "5", "7", "10", "20", "30", "50", "70",
                "100", "125", "150", "175", "200", "225", "250", "300",
                "350", "400", "450", "500", "550", "600", "650", "700",
                "750", "775", "800", "825", "850", "875", "900", "925",
                "950", "975", "1000",
            ],
            "year": date_str[:4],
            "month": date_str[5:7],
            "day": date_str[8:10],
            "time": [f"{h:02d}:00" for h in range(0, 24, 3)],
            "grid": [resolution, resolution],
            "format": "netcdf",
        },
        outfile,
    )
    sz = os.path.getsize(outfile) / 1e6
    print(f"  ✓ {os.path.basename(outfile)} ({sz:.1f} MB)")
    return outfile


def main():
    args = parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
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
    print(f"ERA5 Surface Fields Downloader")
    print(f"  Period:     {args.start} to {args.end}")
    print(f"  Resolution: {args.resolution}°")
    print(f"  Output:     {outdir}")
    print()

    n_days = (end - start).days + 1
    for i, dt in enumerate(date_range(start, end)):
        date_str = dt.strftime("%Y-%m-%d")
        print(f"[{i+1}/{n_days}] {date_str}")
        download_surface_fields(client, date_str, args.resolution, outdir)
        download_convective_fields(client, date_str, args.resolution, outdir)

    print(f"\nDone! Downloaded {n_days} days to {outdir}")


if __name__ == "__main__":
    main()
