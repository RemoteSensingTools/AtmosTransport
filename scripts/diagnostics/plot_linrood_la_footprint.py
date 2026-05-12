#!/usr/bin/env python3
"""Plot the 6-hour LinRood LA footprint from the Julia-exported CSV
(lon/lat computed by AtmosTransport so they match the production CS
geometry exactly)."""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "linrood_la_footprint_c24_6h.csv"


def main():
    if not CSV_PATH.exists():
        sys.exit(f"CSV missing: {CSV_PATH}\n"
                 f"Run scripts/diagnostics/export_linrood_la_footprint_csv.jl first.")

    meta = {}
    rows = []
    with CSV_PATH.open() as f:
        header = None
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                for kv in line.lstrip("# ").split():
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        meta[k] = int(v)
                continue
            if header is None:
                header = line.split(",")
                continue
            parts = line.split(",")
            rows.append((
                int(parts[0]), int(parts[1]), int(parts[2]),
                float(parts[3]), float(parts[4]), float(parts[5]),
            ))

    arr = np.array(rows, dtype=[
        ("panel", "i8"), ("i", "i8"), ("j", "i8"),
        ("lon", "f8"), ("lat", "f8"), ("v", "f8"),
    ])

    Nc = meta.get("Nc", 24)
    nsteps = meta.get("nsteps", 48)
    la_panel = meta.get("la_panel", 4)
    la_i = meta.get("la_i", 5)
    la_j = meta.get("la_j", 22)

    print(f"Loaded {len(arr)} cells (C{Nc}, nsteps={nsteps}, LA at panel {la_panel} ({la_i},{la_j}))")

    abs_v = np.abs(arr["v"])
    vmax = float(abs_v.max())
    print(f"Peak |dJ/dE| = {vmax:.4e}")

    # Wrap longitudes to [-180, 180] for plotting.
    lon_w = np.where(arr["lon"] > 180, arr["lon"] - 360, arr["lon"])
    lat = arr["lat"]

    # Log-scale colour, mask very weak cells.
    eps = vmax * 1e-4
    log_v = np.log10(np.maximum(abs_v, eps) / vmax)
    mask = log_v > -3
    print(f"Cells with |dJ/dE| > vmax/1e3: {int(mask.sum())}/{len(arr)}")

    # LA receptor lon/lat — pick the row matching (panel, i, j).
    la_row = arr[(arr["panel"] == la_panel) &
                 (arr["i"] == la_i) & (arr["j"] == la_j)]
    la_lon = la_row["lon"][0]
    la_lat = la_row["lat"][0]
    la_lon_w = la_lon - 360 if la_lon > 180 else la_lon

    # Try Cartopy for coastlines; fall back to plain matplotlib.
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        fig = plt.figure(figsize=(14, 7))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color="0.3")
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.4, color="0.4")
        transform_kw = {"transform": ccrs.PlateCarree()}
        has_cartopy = True
    except ImportError:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude [°E]")
        ax.set_ylabel("Latitude [°N]")
        ax.grid(True, alpha=0.3)
        transform_kw = {}
        has_cartopy = False

    # Plot all cells in faint background so the CS grid is visible.
    ax.scatter(lon_w, lat, s=4, c="0.85", alpha=0.5, edgecolors="none",
               **transform_kw)
    # Then the masked sensitivity cells on top.
    sc = ax.scatter(lon_w[mask], lat[mask], c=log_v[mask],
                    cmap="viridis", vmin=-3, vmax=0,
                    s=40, alpha=0.95, edgecolors="none",
                    **transform_kw)

    # LA receptor marker.
    ax.plot(la_lon_w, la_lat, marker="x", color="red",
            markersize=22, markeredgewidth=3, **transform_kw)
    ax.annotate(f"LA  ({la_lat:.1f}°N, {la_lon_w:.1f}°E)",
                (la_lon_w, la_lat),
                xytext=(la_lon_w - 50, la_lat + 12),
                color="red", fontsize=13, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
                **transform_kw)

    cbar = plt.colorbar(sc, ax=ax, orientation="horizontal",
                        pad=0.08, shrink=0.6)
    cbar.set_label("log₁₀(|dJ/dE| / max)  —  surface emission sensitivity")

    ax.set_title(
        f"6-hour LinRood adjoint footprint (backward)\n"
        f"Receptor: column-mean tracer at LA — C{Nc} CS grid, "
        f"{nsteps} substeps × 450 s = 6.0 h",
        fontsize=12,
    )

    out_path = CSV_PATH.parent / "linrood_la_footprint_c24_6h.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path} ({out_path.stat().st_size / 1024:.1f} KB) [cartopy={has_cartopy}]")


if __name__ == "__main__":
    main()
