#!/usr/bin/env python3
"""Publish corrected LL/RG validation panel plots from snapshot NetCDF files."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset


@dataclass
class LatLonSnapshot:
    lon: np.ndarray
    lat: np.ndarray
    time_hours: np.ndarray
    surface: np.ndarray
    column: np.ndarray


@dataclass
class CellSnapshot:
    lon_cell: np.ndarray
    lat_cell: np.ndarray
    time_hours: np.ndarray
    surface: np.ndarray
    column: np.ndarray


def _center_lon180(lon_deg: np.ndarray) -> np.ndarray:
    return ((lon_deg + 180.0) % 360.0) - 180.0


def load_latlon_snapshot(path: str) -> LatLonSnapshot:
    with Dataset(path) as ds:
        lon = np.array(ds.variables["lon"][:], dtype=np.float64)
        lat = np.array(ds.variables["lat"][:], dtype=np.float64)
        time_hours = np.array(ds.variables["time_hours"][:], dtype=np.float64)
        surface = np.array(ds.variables["co2_surface"][:], dtype=np.float64)
        column = np.array(ds.variables["co2_column_mean"][:], dtype=np.float64)
    return LatLonSnapshot(lon=lon, lat=lat, time_hours=time_hours, surface=surface, column=column)


def load_cell_snapshot(path: str) -> CellSnapshot:
    with Dataset(path) as ds:
        lon_cell = np.array(ds.variables["lon_cell"][:], dtype=np.float64)
        lat_cell = np.array(ds.variables["lat_cell"][:], dtype=np.float64)
        time_hours = np.array(ds.variables["time_hours"][:], dtype=np.float64)
        surface = np.array(ds.variables["co2_surface"][:], dtype=np.float64)
        column = np.array(ds.variables["co2_column_mean"][:], dtype=np.float64)
    return CellSnapshot(
        lon_cell=lon_cell,
        lat_cell=lat_cell,
        time_hours=time_hours,
        surface=surface,
        column=column,
    )


def _plot_latlon_panel(ax, field_t: np.ndarray, lon: np.ndarray, lat: np.ndarray,
                       vmin: float, vmax: float, title: str):
    order = np.argsort(_center_lon180(lon))
    lon_plot = _center_lon180(lon)[order]
    field_plot = field_t[order, :]
    im = ax.pcolormesh(lon_plot, lat, (field_plot * 1e6).T,
                       shading="auto", cmap="RdYlBu_r", vmin=vmin, vmax=vmax,
                       rasterized=True)
    ax.set_title(title)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    return im


def _plot_cell_panel(ax, field_t: np.ndarray, lon_cell: np.ndarray, lat_cell: np.ndarray,
                     vmin: float, vmax: float, title: str):
    ring_lats = np.unique(np.round(lat_cell, decimals=10))
    ring_lats.sort()
    lat_edges = np.empty(ring_lats.size + 1, dtype=np.float64)
    lat_edges[0] = -90.0
    lat_edges[-1] = 90.0
    if ring_lats.size > 1:
        lat_edges[1:-1] = 0.5 * (ring_lats[:-1] + ring_lats[1:])

    im = None
    for j, lat0 in enumerate(ring_lats):
        mask = np.isclose(lat_cell, lat0, atol=1e-10, rtol=0.0)
        lon_ring = np.sort(_center_lon180(lon_cell[mask]))
        field_ring = field_t[mask][np.argsort(_center_lon180(lon_cell[mask]))]
        lon_edges = _periodic_lon_edges(lon_ring)
        lat_lo = lat_edges[j]
        lat_hi = lat_edges[j + 1]
        x = np.vstack([lon_edges, lon_edges])
        y = np.array([
            np.full_like(lon_edges, lat_lo),
            np.full_like(lon_edges, lat_hi),
        ])
        im = ax.pcolormesh(
            x,
            y,
            (field_ring * 1e6)[None, :],
            shading="flat",
            cmap="RdYlBu_r",
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
    ax.set_title(title)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    return im


def _periodic_lon_edges(lon_centers: np.ndarray) -> np.ndarray:
    if lon_centers.size == 1:
        half_width = 180.0
        return np.array([lon_centers[0] - half_width, lon_centers[0] + half_width], dtype=np.float64)

    lon_sorted = np.sort(lon_centers.astype(np.float64, copy=False))
    midpoints = 0.5 * (lon_sorted + np.concatenate([lon_sorted[1:], lon_sorted[:1] + 360.0]))
    return np.concatenate([[midpoints[-1] - 360.0], midpoints])


def _select_time_indices(time_hours: np.ndarray, max_hour: float) -> list[int]:
    selected = [i for i, t in enumerate(time_hours) if t <= max_hour + 1e-9]
    return selected if selected else list(range(len(time_hours)))


def _row_limits(fields: np.ndarray, time_idx: list[int]) -> tuple[float, float]:
    subset = fields[..., time_idx] * 1e6
    return float(np.nanmin(subset)), float(np.nanmax(subset))


def plot_latlon_panels(data: LatLonSnapshot, max_hour: float, outpath: str, title_prefix: str):
    time_idx = _select_time_indices(data.time_hours, max_hour)
    fig, axes = plt.subplots(2, len(time_idx), figsize=(4.1 * len(time_idx), 8.2), constrained_layout=True)
    surf_vmin, surf_vmax = _row_limits(data.surface, time_idx)
    col_vmin, col_vmax = _row_limits(data.column, time_idx)

    surf_im = None
    col_im = None
    for col, ti in enumerate(time_idx):
        hour = data.time_hours[ti]
        surf_im = _plot_latlon_panel(axes[0, col], data.surface[:, :, ti], data.lon, data.lat,
                                     surf_vmin, surf_vmax, f"t = {hour:.0f} h")
        col_im = _plot_latlon_panel(axes[1, col], data.column[:, :, ti], data.lon, data.lat,
                                    col_vmin, col_vmax, f"t = {hour:.0f} h")
        axes[0, col].set_ylabel("Latitude\nSurface" if col == 0 else "")
        axes[1, col].set_ylabel("Latitude\nColumn mean" if col == 0 else "")
        axes[1, col].set_xlabel("Longitude")

    fig.suptitle(f"{title_prefix} snapshots", fontsize=16)
    fig.colorbar(surf_im, ax=axes[0, :], shrink=0.7, label="CO2 surface [ppm]")
    fig.colorbar(col_im, ax=axes[1, :], shrink=0.7, label="CO2 column mean [ppm]")
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_cell_panels(data: CellSnapshot, max_hour: float, outpath: str, title_prefix: str):
    time_idx = _select_time_indices(data.time_hours, max_hour)
    fig, axes = plt.subplots(2, len(time_idx), figsize=(4.1 * len(time_idx), 8.2), constrained_layout=True)
    surf_vmin, surf_vmax = _row_limits(data.surface, time_idx)
    col_vmin, col_vmax = _row_limits(data.column, time_idx)

    surf_im = None
    col_im = None
    for col, ti in enumerate(time_idx):
        hour = data.time_hours[ti]
        surf_im = _plot_cell_panel(axes[0, col], data.surface[:, ti], data.lon_cell, data.lat_cell,
                                   surf_vmin, surf_vmax, f"t = {hour:.0f} h")
        col_im = _plot_cell_panel(axes[1, col], data.column[:, ti], data.lon_cell, data.lat_cell,
                                  col_vmin, col_vmax, f"t = {hour:.0f} h")
        axes[0, col].set_ylabel("Latitude\nSurface" if col == 0 else "")
        axes[1, col].set_ylabel("Latitude\nColumn mean" if col == 0 else "")
        axes[1, col].set_xlabel("Longitude")

    fig.suptitle(f"{title_prefix} snapshots", fontsize=16)
    fig.colorbar(surf_im, ax=axes[0, :], shrink=0.7, label="CO2 surface [ppm]")
    fig.colorbar(col_im, ax=axes[1, :], shrink=0.7, label="CO2 column mean [ppm]")
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Publish corrected LL/RG validation plots")
    parser.add_argument("--ll", required=True, help="LatLon snapshot NetCDF")
    parser.add_argument("--rg", required=True, help="Reduced-Gaussian snapshot NetCDF")
    parser.add_argument("--output-dir", required=True, help="Directory for output PNGs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ll = load_latlon_snapshot(args.ll)
    rg = load_cell_snapshot(args.rg)

    plot_latlon_panels(ll, 24.0, os.path.join(args.output_dir, "ll96x48_catrine_24h.png"), "LatLon upwind")
    plot_latlon_panels(ll, 48.0, os.path.join(args.output_dir, "ll96x48_catrine_48h.png"), "LatLon upwind")
    plot_cell_panels(rg, 24.0, os.path.join(args.output_dir, "rgN24_catrine_24h.png"), "Reduced Gaussian upwind")
    plot_cell_panels(rg, 48.0, os.path.join(args.output_dir, "rgN24_catrine_48h.png"), "Reduced Gaussian upwind")


if __name__ == "__main__":
    main()
