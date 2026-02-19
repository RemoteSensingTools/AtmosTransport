#!/usr/bin/env python3.9
"""
Download ERA5 GRIB data for TM5 validation.

TM5 filename convention (from tmm_mf_ecmwf_tm5.F90):
  {dir}/{class}-{type}-{YYYYMMDD}-{HHMM}-{levs}-{pid}.128-{grid}.gb

  Examples:
    ea-an-20250201-0000-ml137-138.128-T0639.gb   (spectral VO)
    ea-an-20250201-0000-ml137-130.128-N0320.gb   (gridpoint T)
    ea-an-20250201-0000-sfc-134.128-N0320.gb     (surface SP)
    ea-an-19790101-0000-sfc-129.128-N0320.gb     (invariant Z)

Grid naming uses 4-digit formatting: T0639, N0320
Gribcode uses 3-digit pid + '.128': 138.128
"""
import os
import sys
import time as _time
import cdsapi

OUTDIR = "/tmp/tm5_cfranken/METEO"
os.makedirs(OUTDIR, exist_ok=True)

DATES = ["2025-02-01", "2025-02-02", "2025-02-03"]
TIMES = ["00:00:00", "06:00:00", "12:00:00", "18:00:00"]

SPECTRAL_TRUNC = 639
GAUSSIAN_N = 320

c = cdsapi.Client()


def tm5_filename(ec_class, ec_type, date, timekey, levs, param, grid):
    """Build TM5-convention GRIB filename."""
    datestr = date.replace("-", "")
    gribcode = f"{int(param):d}.128"
    return f"{OUTDIR}/{ec_class}-{ec_type}-{datestr}-{timekey}-{levs}-{gribcode}-{grid}.gb"


def download(dataset, request, outfile, label):
    """Download with retry and skip-if-exists."""
    if os.path.exists(outfile):
        sz = os.path.getsize(outfile) / 1e6
        print(f"  SKIP (exists, {sz:.1f} MB): {os.path.basename(outfile)}", flush=True)
        return True
    print(f"  Downloading {label} -> {os.path.basename(outfile)} ...", flush=True)
    for attempt in range(3):
        try:
            c.retrieve(dataset, request, outfile)
            sz = os.path.getsize(outfile) / 1e6
            print(f"  OK ({sz:.1f} MB): {os.path.basename(outfile)}", flush=True)
            return True
        except Exception as e:
            print(f"  ERROR (attempt {attempt+1}/3): {e}", flush=True)
            if attempt < 2:
                _time.sleep(30)
    return False


grid_spectral = f"T{SPECTRAL_TRUNC:04d}"
grid_gaussian = f"N{GAUSSIAN_N:04d}"

# --- 1. Spectral fields on model levels ---
print("=" * 60, flush=True)
print("1. Spectral fields (VO, D, LNSP) on model levels", flush=True)
print("=" * 60, flush=True)
for date in DATES:
    for t in TIMES:
        timekey = t.replace(":", "")[:4]

        for param, code in [("138", "VO"), ("155", "D")]:
            outfile = tm5_filename("ea", "an", date, timekey, "ml137", param, grid_spectral)
            download("reanalysis-era5-complete", {
                "class": "ea",
                "date": date,
                "expver": "1",
                "levelist": "/".join(str(i) for i in range(1, 138)),
                "levtype": "ml",
                "param": param,
                "stream": "oper",
                "time": t,
                "type": "an",
            }, outfile, f"{code} {date} {timekey}")

        outfile = tm5_filename("ea", "an", date, timekey, "ml1", "152", grid_spectral)
        download("reanalysis-era5-complete", {
            "class": "ea",
            "date": date,
            "expver": "1",
            "levelist": "1",
            "levtype": "ml",
            "param": "152",
            "stream": "oper",
            "time": t,
            "type": "an",
        }, outfile, f"LNSP {date} {timekey}")

# --- 2. Gridpoint fields on model levels (T, Q) ---
print("=" * 60, flush=True)
print("2. Gridpoint model-level fields (T, Q)", flush=True)
print("=" * 60, flush=True)
for date in DATES:
    for t in TIMES:
        timekey = t.replace(":", "")[:4]
        for param, code in [("130", "T"), ("133", "Q")]:
            outfile = tm5_filename("ea", "an", date, timekey, "ml137", param, grid_gaussian)
            download("reanalysis-era5-complete", {
                "class": "ea",
                "date": date,
                "expver": "1",
                "levelist": "/".join(str(i) for i in range(1, 138)),
                "levtype": "ml",
                "param": param,
                "stream": "oper",
                "time": t,
                "type": "an",
                "grid": f"N{GAUSSIAN_N}",
            }, outfile, f"{code} {date} {timekey}")

# --- 3. Surface analysis fields (SP) ---
print("=" * 60, flush=True)
print("3. Surface analysis fields (SP)", flush=True)
print("=" * 60, flush=True)
for date in DATES:
    for t in TIMES:
        timekey = t.replace(":", "")[:4]
        outfile = tm5_filename("ea", "an", date, timekey, "sfc", "134", grid_gaussian)
        download("reanalysis-era5-complete", {
            "class": "ea",
            "date": date,
            "expver": "1",
            "levtype": "sfc",
            "param": "134",
            "stream": "oper",
            "time": t,
            "type": "an",
            "grid": f"N{GAUSSIAN_N}",
        }, outfile, f"SP {date} {timekey}")

# --- 4. Invariant fields (Z, LSM) ---
print("=" * 60, flush=True)
print("4. Invariant fields (Z, LSM)", flush=True)
print("=" * 60, flush=True)
for param, code in [("129", "Z"), ("172", "LSM")]:
    outfile = tm5_filename("ea", "an", "1979-01-01", "0000", "sfc", param, grid_gaussian)
    download("reanalysis-era5-complete", {
        "class": "ea",
        "date": "1979-01-01",
        "expver": "1",
        "levtype": "sfc",
        "param": param,
        "stream": "oper",
        "time": "00:00:00",
        "type": "an",
        "grid": f"N{GAUSSIAN_N}",
    }, outfile, f"invariant {code}")

# --- Summary ---
print("\n" + "=" * 60, flush=True)
print("Download summary", flush=True)
print("=" * 60, flush=True)
total_size = 0
n_files = 0
for f in sorted(os.listdir(OUTDIR)):
    if f.endswith('.gb'):
        sz = os.path.getsize(os.path.join(OUTDIR, f))
        total_size += sz
        n_files += 1
        print(f"  {f} ({sz/1e6:.1f} MB)", flush=True)
print(f"\nTotal: {n_files} files, {total_size/1e9:.2f} GB", flush=True)
