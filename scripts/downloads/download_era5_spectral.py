#!/usr/bin/env python3
"""
Download ERA5 spectral data (VO, D, LNSP) for mass-conserving mass flux computation.

Spectral coefficients (spherical harmonics) are required for the TM5-style
mass flux pipeline (Bregman et al., 2003) which guarantees exact mass
conservation. The gridpoint wind approach (~0.9% mass drift) is a stopgap.

Downloads:
  - VO  (138): vorticity, spectral, all model levels
  - D   (155): divergence, spectral, all model levels
  - LNSP (152): log surface pressure, spectral, level 1

Output: GRIB files, one per day per field group:
  {outdir}/era5_spectral_YYYYMMDD_vo_d.gb     (VO + D, all levels, 4 times)
  {outdir}/era5_spectral_YYYYMMDD_lnsp.gb     (LNSP, level 1, 4 times)

Usage:
    python3 scripts/download_era5_spectral.py --start 2023-06-01 --end 2023-06-30
    python3 scripts/download_era5_spectral.py --start 2023-06-01 --end 2023-06-30 --outdir ~/data/metDrivers/era5/spectral_june2023

Configuration:
    MARS:  ~/.ecmwfapirc  (ECMWF computing account — fast)
    CDS:   ~/.cdsapirc    (Copernicus account — slower but open access)
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Download ERA5 spectral data for mass flux computation")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--outdir", default=None, help="Output directory (auto-generated if omitted)")
    p.add_argument("--times", default="00:00:00/06:00:00/12:00:00/18:00:00",
                   help="Analysis times to retrieve")
    p.add_argument("--force-cds", action="store_true", help="Force CDS API even if MARS available")
    return p.parse_args()


def detect_api(force_cds=False):
    """Detect available API: prefer MARS, fall back to CDS."""
    if not force_cds:
        rc = Path.home() / ".ecmwfapirc"
        if rc.exists():
            try:
                import ecmwfapi
                print("  MARS API available (fast retrieval)")
                return "mars"
            except ImportError:
                print("  ecmwf-api-client not installed, falling back to CDS")

    rc = Path.home() / ".cdsapirc"
    if rc.exists():
        try:
            import cdsapi
            print("  CDS API available (slower, queued retrieval)")
            return "cds"
        except ImportError:
            pass

    print("  No API credentials found. Set up ~/.ecmwfapirc or ~/.cdsapirc")
    sys.exit(1)


def download_spectral_mars(date_str, times, outfile_vo_d, outfile_lnsp):
    """Download spectral fields via ECMWF MARS."""
    import ecmwfapi
    server = ecmwfapi.ECMWFService("mars")

    all_levels = "/".join(str(i) for i in range(1, 138))

    if not os.path.exists(outfile_vo_d):
        print(f"  MARS: VO+D spectral, all 137 levels...")
        server.execute(
            {
                "class": "ea",
                "dataset": "era5",
                "expver": "1",
                "stream": "oper",
                "type": "an",
                "levtype": "ml",
                "levelist": all_levels,
                "param": "138/155",
                "date": date_str,
                "time": times,
            },
            outfile_vo_d,
        )
        sz = os.path.getsize(outfile_vo_d) / 1e6
        print(f"    -> {sz:.1f} MB")
    else:
        print(f"  VO+D exists: {outfile_vo_d}")

    if not os.path.exists(outfile_lnsp):
        print(f"  MARS: LNSP spectral, level 1...")
        server.execute(
            {
                "class": "ea",
                "dataset": "era5",
                "expver": "1",
                "stream": "oper",
                "type": "an",
                "levtype": "ml",
                "levelist": "1",
                "param": "152",
                "date": date_str,
                "time": times,
            },
            outfile_lnsp,
        )
        sz = os.path.getsize(outfile_lnsp) / 1e6
        print(f"    -> {sz:.1f} MB")
    else:
        print(f"  LNSP exists: {outfile_lnsp}")


def download_spectral_cds(date_str, times, outfile_vo_d, outfile_lnsp):
    """Download spectral fields via CDS API."""
    import cdsapi
    c = cdsapi.Client()

    all_levels = "/".join(str(i) for i in range(1, 138))

    if not os.path.exists(outfile_vo_d):
        print(f"  CDS: VO+D spectral, all 137 levels...")
        c.retrieve(
            "reanalysis-era5-complete",
            {
                "class": "ea",
                "expver": "1",
                "stream": "oper",
                "type": "an",
                "levtype": "ml",
                "levelist": all_levels,
                "param": "138/155",
                "date": date_str,
                "time": times,
            },
            outfile_vo_d,
        )
        sz = os.path.getsize(outfile_vo_d) / 1e6
        print(f"    -> {sz:.1f} MB")
    else:
        print(f"  VO+D exists: {outfile_vo_d}")

    if not os.path.exists(outfile_lnsp):
        print(f"  CDS: LNSP spectral, level 1...")
        c.retrieve(
            "reanalysis-era5-complete",
            {
                "class": "ea",
                "expver": "1",
                "stream": "oper",
                "type": "an",
                "levtype": "ml",
                "levelist": "1",
                "param": "152",
                "date": date_str,
                "time": times,
            },
            outfile_lnsp,
        )
        sz = os.path.getsize(outfile_lnsp) / 1e6
        print(f"    -> {sz:.1f} MB")
    else:
        print(f"  LNSP exists: {outfile_lnsp}")


def main():
    args = parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    if args.outdir:
        outdir = os.path.expanduser(args.outdir)
    else:
        outdir = os.path.expanduser(
            f"~/data/metDrivers/era5/spectral_{args.start.replace('-', '')}_{args.end.replace('-', '')}"
        )
    os.makedirs(outdir, exist_ok=True)

    n_total = (end - start).days + 1

    print("=" * 60)
    print("ERA5 Spectral Data Download")
    print("=" * 60)
    print(f"  Period:   {args.start} to {args.end} ({n_total} days)")
    print(f"  Times:    {args.times}")
    print(f"  Fields:   VO (138), D (155), LNSP (152)")
    print(f"  Levels:   1-137 (all model levels)")
    print(f"  Format:   GRIB (native spectral, T639)")
    print(f"  Output:   {outdir}")

    # Estimate: T639 spectral ≈ 1.6 MB/field
    #   VO+D: 2 fields × 137 levels × 4 times × 30 days ≈ 52 GB
    #   LNSP: 1 field × 1 level × 4 times × 30 days ≈ 0.2 GB
    est_gb = n_total * (2 * 137 * 4 * 1.6 + 1 * 4 * 1.6) / 1000
    print(f"  Estimated size: ~{est_gb:.0f} GB")
    print("=" * 60)

    api = detect_api(args.force_cds)
    download_fn = download_spectral_mars if api == "mars" else download_spectral_cds

    day = start
    n_done = 0

    while day <= end:
        date_str = day.strftime("%Y-%m-%d")
        day_tag = day.strftime("%Y%m%d")

        outfile_vo_d = os.path.join(outdir, f"era5_spectral_{day_tag}_vo_d.gb")
        outfile_lnsp = os.path.join(outdir, f"era5_spectral_{day_tag}_lnsp.gb")

        print(f"\n[{n_done + 1}/{n_total}] {date_str}")

        # Check if both files exist
        have_vo_d = os.path.exists(outfile_vo_d) and os.path.getsize(outfile_vo_d) > 1e6
        have_lnsp = os.path.exists(outfile_lnsp) and os.path.getsize(outfile_lnsp) > 1e3
        if have_vo_d and have_lnsp:
            print(f"  Already complete")
        else:
            try:
                download_fn(date_str, args.times, outfile_vo_d, outfile_lnsp)
            except Exception as e:
                print(f"  ERROR: {e}")
                print(f"  Continuing to next day...")

        day += timedelta(days=1)
        n_done += 1

    # Summary
    print(f"\n{'=' * 60}")
    print("Download summary")
    print("=" * 60)
    total_size = 0
    n_files = 0
    for f in sorted(os.listdir(outdir)):
        if f.endswith('.gb'):
            sz = os.path.getsize(os.path.join(outdir, f))
            total_size += sz
            n_files += 1
    print(f"  {n_files} files, {total_size / 1e9:.1f} GB total")
    print(f"  Directory: {outdir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
