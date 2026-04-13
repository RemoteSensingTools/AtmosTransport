#!/usr/bin/env python3
"""
Download ERA5 model-level data via ECMWF MARS (fast) or CDS API (fallback).

Variables: U (131), V (132), W/omega (135), LNSP (152)
Output: one NetCDF file per day with all variables merged.

MARS is ~10-100x faster than CDS for model-level data because it
retrieves directly from ECMWF's archive without CDS queue overhead.

Usage:
    python3 scripts/download_era5_mars.py --start 2024-06-01 --end 2024-06-07
    python3 scripts/download_era5_mars.py --start 2024-06-01 --end 2024-06-07 --resolution 1.0

Configuration:
    MARS:  ~/.ecmwfapirc  (ECMWF computing account required)
    CDS:   ~/.cdsapirc    (Copernicus account, slower but open access)

The script auto-detects which API is available and prefers MARS.
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Download ERA5 model-level data")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--resolution", type=float, default=1.0, help="Grid spacing in degrees")
    p.add_argument("--level-top", type=int, default=50, help="Topmost model level")
    p.add_argument("--level-bot", type=int, default=137, help="Bottommost model level")
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
                print("✓ MARS API available (fast retrieval)")
                return "mars"
            except ImportError:
                print("  ecmwf-api-client not installed, falling back to CDS")

    rc = Path.home() / ".cdsapirc"
    if rc.exists():
        try:
            import cdsapi
            print("✓ CDS API available (slower, queued retrieval)")
            return "cds"
        except ImportError:
            pass

    print("✗ No API credentials found. Set up ~/.ecmwfapirc or ~/.cdsapirc")
    sys.exit(1)


def download_mars(date_str, levelist, resolution, times, outfile_ml, outfile_lnsp):
    """Retrieve via ECMWF MARS — fast, direct archive access."""
    import ecmwfapi
    server = ecmwfapi.ECMWFService("mars")

    if not os.path.exists(outfile_ml):
        print(f"  MARS: retrieving ML fields for {date_str}...")
        server.execute(
            {
                "class": "ea",
                "dataset": "era5",
                "expver": "1",
                "stream": "oper",
                "type": "an",
                "levtype": "ml",
                "levelist": levelist,
                "param": "131/132/135",
                "date": date_str,
                "time": times,
                "grid": f"{resolution}/{resolution}",
                "format": "netcdf",
            },
            outfile_ml,
        )
        print(f"    → {os.path.getsize(outfile_ml)/1e6:.1f} MB")
    else:
        print(f"  ML already exists: {outfile_ml}")

    if not os.path.exists(outfile_lnsp):
        print(f"  MARS: retrieving LNSP for {date_str}...")
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
                "grid": f"{resolution}/{resolution}",
                "format": "netcdf",
            },
            outfile_lnsp,
        )
        print(f"    → {os.path.getsize(outfile_lnsp)/1e6:.1f} MB")
    else:
        print(f"  LNSP already exists: {outfile_lnsp}")


def download_cds(date_str, levelist, resolution, times, outfile_ml, outfile_lnsp):
    """Retrieve via CDS API — slower but universally accessible."""
    import cdsapi
    c = cdsapi.Client()

    if not os.path.exists(outfile_ml):
        print(f"  CDS: retrieving ML fields for {date_str}...")
        c.retrieve(
            "reanalysis-era5-complete",
            {
                "class": "ea",
                "expver": "1",
                "stream": "oper",
                "type": "an",
                "levtype": "ml",
                "levelist": levelist,
                "param": "131/132/135",
                "date": date_str,
                "time": times,
                "grid": f"{resolution}/{resolution}",
                "format": "netcdf",
            },
            outfile_ml,
        )
        print(f"    → {os.path.getsize(outfile_ml)/1e6:.1f} MB")
    else:
        print(f"  ML already exists: {outfile_ml}")

    if not os.path.exists(outfile_lnsp):
        print(f"  CDS: retrieving LNSP for {date_str}...")
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
                "grid": f"{resolution}/{resolution}",
                "format": "netcdf",
            },
            outfile_lnsp,
        )
        print(f"    → {os.path.getsize(outfile_lnsp)/1e6:.1f} MB")
    else:
        print(f"  LNSP already exists: {outfile_lnsp}")


def _copy_var(src_var, dst_ds, dst_dims=None, dst_name=None):
    """Copy a netCDF variable handling _FillValue correctly."""
    import numpy as np
    name = dst_name or src_var.name
    dims = dst_dims or src_var.dimensions
    attrs = {k: src_var.getncattr(k) for k in src_var.ncattrs()}
    fill = attrs.pop("_FillValue", None)
    outvar = dst_ds.createVariable(name, src_var.datatype, dims,
                                   zlib=True, complevel=4, fill_value=fill)
    outvar.setncatts(attrs)
    return outvar


def merge_ml_lnsp(ml_file, lnsp_file, outfile):
    """Merge ML (u,v,w) and LNSP into a single NetCDF file."""
    import netCDF4 as nc
    import numpy as np

    if os.path.exists(outfile):
        print(f"  Combined file exists: {outfile}")
        return

    print(f"  Merging ML + LNSP → {os.path.basename(outfile)}")
    with nc.Dataset(ml_file) as src_ml, nc.Dataset(lnsp_file) as src_lnsp:
        with nc.Dataset(outfile, "w", format="NETCDF4") as dst:
            for dname, dim in src_ml.dimensions.items():
                dst.createDimension(dname, len(dim) if not dim.isunlimited() else None)

            for vname, varin in src_ml.variables.items():
                outvar = _copy_var(varin, dst)
                outvar[:] = varin[:]

            lnsp_name = None
            for k in src_lnsp.variables:
                if k.lower() == "lnsp":
                    lnsp_name = k
                    break

            if lnsp_name:
                lnsp_var = src_lnsp.variables[lnsp_name]
                lnsp_data = lnsp_var[:]

                lnsp_dims = list(lnsp_var.dimensions)
                level_dims = [d for d in lnsp_dims if d in ("level", "model_level")]
                if level_dims:
                    ax = lnsp_dims.index(level_dims[0])
                    lnsp_data = np.squeeze(lnsp_data, axis=ax)
                    lnsp_dims.remove(level_dims[0])

                for d in lnsp_dims:
                    if d not in dst.dimensions:
                        dim_size = src_lnsp.dimensions[d].size
                        dst.createDimension(d, None if src_lnsp.dimensions[d].isunlimited() else dim_size)

                outvar = _copy_var(lnsp_var, dst, dst_dims=lnsp_dims, dst_name="lnsp")
                outvar[:] = lnsp_data
            else:
                print(f"  WARNING: LNSP variable not found in {lnsp_file}")

    size_mb = os.path.getsize(outfile) / 1e6
    print(f"    → {size_mb:.1f} MB (compressed)")


def main():
    args = parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    res_str = str(args.resolution).replace(".", "")

    if args.outdir:
        outdir = args.outdir
    else:
        outdir = os.path.expanduser(
            f"~/data/metDrivers/era5/era5_ml_{res_str}deg_{args.start.replace('-','')}_{args.end.replace('-','')}"
        )
    os.makedirs(outdir, exist_ok=True)

    levelist = "/".join(str(l) for l in range(args.level_top, args.level_bot + 1))
    n_levels = args.level_bot - args.level_top + 1

    print("=" * 60)
    print("ERA5 Model-Level Download")
    print("=" * 60)
    print(f"  Resolution: {args.resolution}°")
    print(f"  Period:     {args.start} to {args.end}")
    print(f"  Levels:     {args.level_top}-{args.level_bot} ({n_levels} levels)")
    print(f"  Times:      {args.times}")
    print(f"  Output:     {outdir}")

    api = detect_api(args.force_cds)
    download_fn = download_mars if api == "mars" else download_cds

    day = start
    n_done = 0
    n_total = (end - start).days + 1

    while day <= end:
        date_str = day.strftime("%Y-%m-%d")
        day_tag = day.strftime("%Y%m%d")

        final_file = os.path.join(outdir, f"era5_ml_{day_tag}.nc")
        ml_tmp = final_file + ".ml_tmp"
        lnsp_tmp = final_file + ".lnsp_tmp"

        print(f"\n[{n_done+1}/{n_total}] {date_str}")

        if os.path.exists(final_file):
            print(f"  Already complete: {final_file}")
        else:
            try:
                download_fn(date_str, levelist, args.resolution, args.times, ml_tmp, lnsp_tmp)
                merge_ml_lnsp(ml_tmp, lnsp_tmp, final_file)

                # Clean up temp files only after successful merge
                if os.path.exists(final_file) and os.path.getsize(final_file) > 1e6:
                    for tmp in (ml_tmp, lnsp_tmp):
                        if os.path.exists(tmp):
                            os.remove(tmp)
                    print(f"  ✓ Complete")
                else:
                    print(f"  ✗ Output file too small, keeping temp files")
            except Exception as e:
                print(f"  ✗ Error: {e}")
                print(f"    Temp files preserved for retry")

        day += timedelta(days=1)
        n_done += 1

    print(f"\n{'=' * 60}")
    print(f"Download complete: {n_done} days processed")
    print(f"Files in {outdir}:")
    for f in sorted(os.listdir(outdir)):
        if f.endswith(".nc") and not f.endswith("_tmp"):
            sz = os.path.getsize(os.path.join(outdir, f)) / 1e6
            print(f"  {f}  ({sz:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
