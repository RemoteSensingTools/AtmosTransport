#!/usr/bin/env python3
"""
Generate PNG snapshots for 8-way Catrine 2-day comparison.

Reads snapshot NetCDFs from ~/data/AtmosTransport/output/catrine_2day/
and produces PNG panels for each snapshot hour.

Usage:
    python3 scripts/visualization/plot_catrine_8way_snapshots.py [--outdir <dir>]
"""

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from netCDF4 import Dataset

DATA_DIR = os.path.expanduser("~/data/AtmosTransport/output/catrine_2day")
OUT_DIR = os.path.expanduser("~/data/AtmosTransport/output/catrine_2day/png")

# Config: (label, filename, grid_type)
CONFIGS = [
    ("LL Upwind",   "ll_upwind.nc",   "structured"),
    ("LL Slopes",   "ll_slopes.nc",   "structured"),
    ("LL PPM",      "ll_ppm.nc",      "structured"),
    ("RG Upwind",   "rg_upwind.nc",   "unstructured"),
    ("CS Upwind",   "cs_upwind.nc",   "unstructured"),
    ("CS Slopes",   "cs_slopes.nc",   "unstructured"),
    ("CS PPM",      "cs_ppm.nc",      "unstructured"),
    ("CS LinRood",  "cs_linrood.nc",  "unstructured"),
]


def load_snapshot(path):
    """Load a snapshot NetCDF and return (lons, lats, times, tracer_data)."""
    ds = Dataset(path, "r")
    times = ds.variables.get("time_hours", ds.variables.get("time", None))
    if times is None:
        ds.close()
        return None
    times = np.array(times[:])

    # Find lon/lat variables — handle GCHP-compatible (Xdim, Ydim, nf) and legacy flat formats
    is_cs_structured = "lons" in ds.variables and "nf" in ds.dimensions
    if is_cs_structured:
        # GCHP-compatible: lons(Xdim, Ydim, nf) → flatten to 1D
        lons = np.array(ds.variables["lons"][:]).ravel(order="F")  # column-major (Julia)
        lats = np.array(ds.variables["lats"][:]).ravel(order="F")
    elif "lon_cell" in ds.variables:
        lons = np.array(ds.variables["lon_cell"][:])
        lats = np.array(ds.variables["lat_cell"][:])
    elif "lon" in ds.dimensions:
        lon = np.array(ds.variables["lon"][:])
        lat = np.array(ds.variables["lat"][:])
        lons, lats = np.meshgrid(lon, lat)
        lons = lons.ravel()
        lats = lats.ravel()
    else:
        ds.close()
        return None

    # Collect tracer data
    tracers = {}
    for vname in ds.variables:
        if vname.endswith("_column_mean"):
            tname = vname.replace("_column_mean", "")
            data = np.array(ds.variables[vname][:])
            if is_cs_structured and data.ndim == 4:
                # (Xdim, Ydim, nf, time) → (cell, time) with column-major panel flatten
                Nx, Ny, Npanel, Nt = data.shape
                flat = data.reshape(Nx * Ny * Npanel, Nt, order="F")
                tracers[tname] = flat
            elif data.ndim == 2:
                # Ensure shape is (cell, time) — may come as (time, cell)
                if data.shape[0] == len(lons):
                    tracers[tname] = data  # already (cell, time)
                else:
                    tracers[tname] = data.T  # transpose (time, cell) → (cell, time)
            elif data.ndim == 1:
                tracers[tname] = data[:, np.newaxis]
    ds.close()

    if not tracers:
        return None
    return lons, lats, times, tracers


def plot_snapshot_hour(configs_data, hour_idx, hour, tracer_name, outdir):
    """Plot one hour for one tracer across all configs."""
    n_configs = len(configs_data)
    ncols = 4
    nrows = (n_configs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows),
                              subplot_kw={"aspect": "auto"})
    if nrows == 1:
        axes = axes[np.newaxis, :]

    # Compute global color range across all configs for this hour
    vmin, vmax = np.inf, -np.inf
    for label, data in configs_data:
        if data is None:
            continue
        lons, lats, times, tracers = data
        if tracer_name not in tracers:
            continue
        vals = tracers[tracer_name]
        if hour_idx < vals.shape[1]:
            v = vals[:, hour_idx] * 1e6  # convert to ppm
            v = v[np.isfinite(v)]
            if len(v) > 0:
                vmin = min(vmin, np.nanpercentile(v, 2))
                vmax = max(vmax, np.nanpercentile(v, 98))

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin, vmax = 0, 1

    norm = Normalize(vmin=vmin, vmax=vmax)

    for idx, (label, data) in enumerate(configs_data):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        if data is None:
            ax.text(0.5, 0.5, f"{label}\n(no data)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.set_xlim(0, 360)
            ax.set_ylim(-90, 90)
            ax.set_title(label, fontsize=11, fontweight="bold")
            continue

        lons, lats, times, tracers = data
        if tracer_name not in tracers or hour_idx >= tracers[tracer_name].shape[1]:
            ax.text(0.5, 0.5, f"{label}\n(no tracer)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(label, fontsize=11, fontweight="bold")
            continue

        vals = tracers[tracer_name][:, hour_idx] * 1e6  # ppm
        # Handle NaN/Inf: replace with 0 for plotting
        vals = np.where(np.isfinite(vals), vals, 0.0)
        sc = ax.scatter(lons, lats, c=vals, s=0.3, norm=norm, cmap="RdYlBu_r",
                        rasterized=True)
        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    # Hide empty axes
    for idx in range(n_configs, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(f"{tracer_name} column mean [ppm] — t = {hour:.0f}h",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="RdYlBu_r"),
                 ax=axes.ravel().tolist(), shrink=0.6, label="ppm", pad=0.02)

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{tracer_name}_t{hour:04.0f}h.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Catrine 8-way comparison snapshots")
    parser.add_argument("--datadir", default=DATA_DIR, help="Input data directory")
    parser.add_argument("--outdir", default=OUT_DIR, help="Output PNG directory")
    args = parser.parse_args()

    # Load all configs
    configs_data = []
    for label, fname, gtype in CONFIGS:
        path = os.path.join(args.datadir, fname)
        if os.path.exists(path):
            data = load_snapshot(path)
            configs_data.append((label, data))
            print(f"Loaded: {label} ({path})")
        else:
            configs_data.append((label, None))
            print(f"Missing: {label} ({path})")

    # Determine snapshot hours from first available config
    hours = None
    for label, data in configs_data:
        if data is not None:
            hours = data[2]
            break

    if hours is None:
        print("No snapshot data found. Run the simulations first.")
        sys.exit(1)

    # Find available tracers
    tracer_names = set()
    for label, data in configs_data:
        if data is not None:
            tracer_names.update(data[3].keys())

    print(f"\nSnapshot hours: {hours}")
    print(f"Tracers: {tracer_names}")
    print(f"Output: {args.outdir}")

    for tracer in sorted(tracer_names):
        print(f"\nPlotting {tracer}...")
        for hi, hour in enumerate(hours):
            plot_snapshot_hour(configs_data, hi, hour, tracer, args.outdir)

    print(f"\nDone. PNGs in {args.outdir}")


if __name__ == "__main__":
    main()
