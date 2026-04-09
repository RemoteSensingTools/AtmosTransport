# Download Script Notes

This directory still contains a mix of current entrypoints and older
ERA5-specific download helpers.

## Current ERA5 entrypoints

- `download_era5_native_monthly.py`
  Canonical native ERA5 monthly GRIB downloader for the new
  `~/data/AtmosTransport/met/era5/N320/hourly/raw/...` layout.
- `download_era5_physics.py`
  Current daily lat-lon NetCDF staging path for the existing ERA5
  preprocessing pipeline.
- `download_era5_grib_tm5.py`
  TM5-oriented native GRIB pull with TM5 filename conventions.

## Backup snapshot

Older or transitional ERA5 download helpers are copied into:

- `scripts/downloads/bck/era5_legacy/`

These copies are intentionally redundant for now. The goal is to make it easy
to prune the main directory later without losing historical entrypoints while
the docs and preprocessing scripts are still being updated.
