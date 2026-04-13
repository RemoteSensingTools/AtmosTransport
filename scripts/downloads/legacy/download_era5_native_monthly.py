#!/usr/bin/env python3
"""
Download monthly native ERA5 GRIB bundles into the canonical AtmosTransport tree.

This script keeps ERA5 in native GRIB form and separates requests by archive
family:

  1. Model-level analyses ("core"):
     - 130: temperature            (T639 native spectral)
     - 133: specific humidity      (N320 reduced Gaussian)
     - 138: vorticity              (T639 native spectral)
     - 152: log surface pressure   (T639 native spectral; level 1 only)
     - 155: divergence             (T639 native spectral)

  2. Model-level convection forecasts:
     - 235009: time-mean updraught mass flux
     - 235010: time-mean downdraught mass flux
     - 235011: time-mean updraught detrainment rate
     - 235012: time-mean downdraught detrainment rate

  3. Single-level surface analyses:
     - surface pressure, geopotential, land-sea mask
     - boundary layer height, friction velocity
     - 10m winds, 2m temperature, 2m dewpoint
     - sensible and latent heat flux

The output tree follows the canonical raw native layout:

  ~/data/AtmosTransport/met/era5/N320/hourly/raw/ml_an_native_core/
  ~/data/AtmosTransport/met/era5/N320/hourly/raw/ml_fc_convection/
  ~/data/AtmosTransport/met/era5/N320/hourly/raw/sfc_an_native/

Example:
    python3 scripts/downloads/download_era5_native_monthly.py \
        --year 2021 --month 12
"""

import argparse
import calendar
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List


DEFAULT_ROOT = Path("~/data/AtmosTransport/met/era5/N320/hourly/raw").expanduser()
RETRY_WAIT_SECONDS = 120
MIN_VALID_BYTES = 1024


@dataclass(frozen=True)
class DownloadBundle:
    key: str
    dataset: str
    subdir: str
    filename_template: str
    description: str
    request_builder: Callable[[int, int], Dict[str, object]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download monthly native ERA5 GRIB bundles"
    )
    p.add_argument("--year", type=int, required=True, help="Year, e.g. 2021")
    p.add_argument("--month", type=int, required=True, help="Month number, e.g. 12")
    p.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help=f"Output root (default: {DEFAULT_ROOT})",
    )
    p.add_argument(
        "--bundles",
        default="all",
        help="Comma-separated bundle names: all, core, convection, surface",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Maximum download attempts per bundle (default: 5)",
    )
    p.add_argument(
        "--retry-wait-seconds",
        type=int,
        default=RETRY_WAIT_SECONDS,
        help=f"Delay between retries in seconds (default: {RETRY_WAIT_SECONDS})",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download even if the target file already exists",
    )
    return p.parse_args()


def require_cdsapi() -> None:
    try:
        import cdsapi  # noqa: F401
    except ImportError:
        print("ERROR: cdsapi not installed. Run: pip install 'cdsapi>=0.7.7'", file=sys.stderr)
        sys.exit(1)

    rc = Path.home() / ".cdsapirc"
    if not rc.exists():
        print("ERROR: ~/.cdsapirc not found. Configure CDS credentials first.", file=sys.stderr)
        sys.exit(1)


def month_tag(year: int, month: int) -> str:
    return f"{year}{month:02d}"


def month_days(year: int, month: int) -> List[str]:
    ndays = calendar.monthrange(year, month)[1]
    return [f"{day:02d}" for day in range(1, ndays + 1)]


def month_date_range(year: int, month: int) -> str:
    ndays = calendar.monthrange(year, month)[1]
    return f"{year}-{month:02d}-01/to/{year}-{month:02d}-{ndays:02d}"


def hourly_times() -> str:
    return "/".join(f"{hour:02d}:00:00" for hour in range(24))


def build_core_request(year: int, month: int) -> Dict[str, object]:
    return {
        "class": "ea",
        "expver": "1",
        "stream": "oper",
        "type": "an",
        "levtype": "ml",
        "levelist": "1/to/137",
        "param": "130/133/138/152/155",
        "date": month_date_range(year, month),
        "time": hourly_times(),
        "format": "grib",
    }


def build_convection_request(year: int, month: int) -> Dict[str, object]:
    return {
        "class": "ea",
        "expver": "1",
        "stream": "oper",
        "type": "fc",
        "levtype": "ml",
        "levelist": "1/to/137",
        "param": "235009/235010/235011/235012",
        "date": month_date_range(year, month),
        "time": "06:00:00/18:00:00",
        "step": "/".join(str(step) for step in range(1, 13)),
        "format": "grib",
    }


def build_surface_request(year: int, month: int) -> Dict[str, object]:
    return {
        "product_type": ["reanalysis"],
        "variable": [
            "surface_pressure",
            "geopotential",
            "land_sea_mask",
            "boundary_layer_height",
            "friction_velocity",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "2m_dewpoint_temperature",
            "surface_sensible_heat_flux",
            "surface_latent_heat_flux",
        ],
        "year": [f"{year}"],
        "month": [f"{month:02d}"],
        "day": month_days(year, month),
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "data_format": "grib",
        "download_format": "unarchived",
    }


BUNDLES: Dict[str, DownloadBundle] = {
    "core": DownloadBundle(
        key="core",
        dataset="reanalysis-era5-complete",
        subdir="ml_an_native_core",
        filename_template="era5_ml_an_native_core_{tag}.grib",
        description="Model-level analyses: T, Q, VO, LNSP, D",
        request_builder=build_core_request,
    ),
    "convection": DownloadBundle(
        key="convection",
        dataset="reanalysis-era5-complete",
        subdir="ml_fc_convection",
        filename_template="era5_ml_fc_convection_{tag}.grib",
        description="Model-level forecast convection: UDMF, DDMF, UDRF, DDRF",
        request_builder=build_convection_request,
    ),
    "surface": DownloadBundle(
        key="surface",
        dataset="reanalysis-era5-single-levels",
        subdir="sfc_an_native",
        filename_template="era5_sfc_an_native_{tag}.grib",
        description="Single-level analyses: surface pressure, PBL, fluxes, 2m/10m fields",
        request_builder=build_surface_request,
    ),
}


def selected_bundles(spec: str) -> List[DownloadBundle]:
    tokens = {token.strip().lower() for token in spec.split(",") if token.strip()}
    if not tokens or "all" in tokens:
        return [BUNDLES["core"], BUNDLES["convection"], BUNDLES["surface"]]

    unknown = sorted(tokens - set(BUNDLES))
    if unknown:
        raise SystemExit(
            f"Unknown bundle(s): {', '.join(unknown)}. Valid bundles: all, core, convection, surface"
        )

    return [BUNDLES[name] for name in ("core", "convection", "surface") if name in tokens]


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def valid_existing_file(path: Path) -> bool:
    return path.exists() and path.stat().st_size >= MIN_VALID_BYTES


def retrieve_with_retries(
    client,
    bundle: DownloadBundle,
    request: Dict[str, object],
    target: Path,
    retries: int,
    retry_wait_seconds: int,
) -> None:
    tmp_target = target.with_suffix(target.suffix + ".part")

    if tmp_target.exists():
        tmp_target.unlink()

    for attempt in range(1, retries + 1):
        try:
            print(f"  Requesting {bundle.key} (attempt {attempt}/{retries})", flush=True)
            client.retrieve(bundle.dataset, request, str(tmp_target))
            if not valid_existing_file(tmp_target):
                raise RuntimeError(f"incomplete download: {tmp_target}")
            tmp_target.replace(target)
            size_gb = target.stat().st_size / 1e9
            print(f"  Saved {target.name} ({size_gb:.2f} GB)", flush=True)
            return
        except Exception as exc:
            print(f"  ERROR downloading {bundle.key}: {exc}", flush=True)
            if tmp_target.exists():
                tmp_target.unlink()
            if attempt >= retries:
                raise
            print(f"  Waiting {retry_wait_seconds}s before retry...", flush=True)
            time.sleep(retry_wait_seconds)


def download_bundle(
    client,
    root: Path,
    bundle: DownloadBundle,
    year: int,
    month: int,
    retries: int,
    retry_wait_seconds: int,
    overwrite: bool,
) -> Path:
    tag = month_tag(year, month)
    outdir = root / bundle.subdir
    ensure_directory(outdir)
    target = outdir / bundle.filename_template.format(tag=tag)

    print(f"\n[{bundle.key}] {bundle.description}", flush=True)
    print(f"  Dataset: {bundle.dataset}", flush=True)
    print(f"  Target:  {target}", flush=True)

    if valid_existing_file(target) and not overwrite:
        size_gb = target.stat().st_size / 1e9
        print(f"  SKIP existing file ({size_gb:.2f} GB)", flush=True)
        return target

    request = bundle.request_builder(year, month)
    retrieve_with_retries(
        client,
        bundle,
        request,
        target,
        retries=retries,
        retry_wait_seconds=retry_wait_seconds,
    )
    return target


def main() -> None:
    args = parse_args()
    if not (1 <= args.month <= 12):
        raise SystemExit("--month must be between 1 and 12")

    require_cdsapi()
    import cdsapi

    bundles = selected_bundles(args.bundles)
    root = Path(args.root).expanduser()
    ensure_directory(root)

    print("=" * 72, flush=True)
    print("ERA5 Native Monthly Downloader", flush=True)
    print("=" * 72, flush=True)
    print(f"  Month:      {args.year}-{args.month:02d}", flush=True)
    print(f"  Root:       {root}", flush=True)
    print(f"  Bundles:    {', '.join(bundle.key for bundle in bundles)}", flush=True)
    print(f"  Overwrite:  {args.overwrite}", flush=True)
    print("=" * 72, flush=True)

    client = cdsapi.Client()

    completed: List[Path] = []
    for bundle in bundles:
        completed.append(
            download_bundle(
                client,
                root=root,
                bundle=bundle,
                year=args.year,
                month=args.month,
                retries=args.retries,
                retry_wait_seconds=args.retry_wait_seconds,
                overwrite=args.overwrite,
            )
        )

    print("\nCompleted bundles:", flush=True)
    for path in completed:
        print(f"  - {path}", flush=True)


if __name__ == "__main__":
    main()
