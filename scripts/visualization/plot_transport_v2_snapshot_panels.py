#!/usr/bin/env python3
"""Panel plotter for src_v2 snapshot NetCDF files."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np


def _plot_latlon(ax, lon, lat, data, vmin, vmax):
    im = ax.pcolormesh(lon, lat, data, shading="auto", cmap="RdYlBu_r", vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_xlim(float(np.min(lon)), float(np.max(lon)))
    ax.set_ylim(float(np.min(lat)), float(np.max(lat)))
    return im


def _plot_reduced(ax, lon, lat, data, vmin, vmax):
    im = ax.scatter(lon, lat, c=data, s=1.2, marker="s", linewidths=0, cmap="RdYlBu_r", vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_xlim(0.0, 360.0)
    ax.set_ylim(-90.0, 90.0)
    return im


def _latlon_slice(var, i):
    dims = var.dimensions
    data = var[:]
    if dims == ("time", "lat", "lon"):
        return np.asarray(data[i, :, :])
    if dims == ("lat", "lon", "time"):
        return np.asarray(data[:, :, i])
    if dims == ("lon", "lat", "time"):
        return np.asarray(data[:, :, i]).T
    if dims == ("time", "lon", "lat"):
        return np.asarray(data[i, :, :]).T
    raise ValueError(f"Unsupported latlon dims {dims!r}")


def _reduced_slice(var, i):
    dims = var.dimensions
    data = var[:]
    if dims == ("time", "cell"):
        return np.asarray(data[i, :])
    if dims == ("cell", "time"):
        return np.asarray(data[:, i])
    raise ValueError(f"Unsupported reduced dims {dims!r}")


def _all_slices(var, grid_type):
    ntime = var.shape[0] if var.dimensions[0] == "time" else var.shape[-1]
    if grid_type == "latlon":
        return np.stack([_latlon_slice(var, i) for i in range(ntime)], axis=0)
    return np.stack([_reduced_slice(var, i) for i in range(ntime)], axis=0)


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 scripts/visualization/plot_transport_v2_snapshot_panels.py <input.nc> <output.png>")
        sys.exit(1)

    in_path = Path(sys.argv[1]).expanduser()
    out_path = Path(sys.argv[2]).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = nc.Dataset(in_path, "r")
    try:
        grid_type = getattr(ds, "grid_type")
        scheme = getattr(ds, "scheme", "unknown")
        tracer_name = getattr(ds, "tracer_name", "tracer")
        times = np.asarray(ds.variables["time_hours"][:])
        sfc_var = ds.variables["co2_surface"]
        col_var = ds.variables["co2_column_mean"]
        sfc_all = _all_slices(sfc_var, grid_type)
        col_all = _all_slices(col_var, grid_type)

        sfc_vmin, sfc_vmax = np.nanpercentile(sfc_all, [1, 99])
        col_vmin, col_vmax = np.nanpercentile(col_all, [1, 99])

        ncols = len(times)
        fig, axs = plt.subplots(2, ncols, figsize=(3.6 * ncols, 7.2), constrained_layout=True)
        if ncols == 1:
            axs = np.array(axs).reshape(2, 1)

        if grid_type == "latlon":
            lon = np.asarray(ds.variables["lon"][:])
            lat = np.asarray(ds.variables["lat"][:])
            for i, hour in enumerate(times):
                im1 = _plot_latlon(axs[0, i], lon, lat, sfc_all[i], sfc_vmin, sfc_vmax)
                im2 = _plot_latlon(axs[1, i], lon, lat, col_all[i], col_vmin, col_vmax)
                axs[0, i].set_title(f"t = {hour:g} h")
                axs[1, i].set_xlabel("Longitude")
            axs[0, 0].set_ylabel("Latitude\nSurface")
            axs[1, 0].set_ylabel("Latitude\nColumn mean")
        elif grid_type == "reduced_gaussian":
            lon = np.asarray(ds.variables["lon_cell"][:])
            lat = np.asarray(ds.variables["lat_cell"][:])
            for i, hour in enumerate(times):
                im1 = _plot_reduced(axs[0, i], lon, lat, sfc_all[i], sfc_vmin, sfc_vmax)
                im2 = _plot_reduced(axs[1, i], lon, lat, col_all[i], col_vmin, col_vmax)
                axs[0, i].set_title(f"t = {hour:g} h")
                axs[1, i].set_xlabel("Longitude")
            axs[0, 0].set_ylabel("Latitude\nSurface")
            axs[1, 0].set_ylabel("Latitude\nColumn mean")
        else:
            raise ValueError(f"Unsupported grid_type {grid_type!r}")

        cbar1 = fig.colorbar(im1, ax=axs[0, :], orientation="horizontal", fraction=0.05, pad=0.08)
        cbar1.set_label(f"{tracer_name} surface mixing ratio")
        cbar2 = fig.colorbar(im2, ax=axs[1, :], orientation="horizontal", fraction=0.05, pad=0.08)
        cbar2.set_label(f"{tracer_name} column-mean mixing ratio")

        fig.suptitle(f"{grid_type.replace('_', ' ').title()} {scheme} snapshots", fontsize=14, y=1.02)
        fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white")
        print(f"Saved {out_path}")
    finally:
        ds.close()


if __name__ == "__main__":
    main()
