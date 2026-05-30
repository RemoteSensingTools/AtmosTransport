#!/usr/bin/env python3
"""CATRINE C180 map + longitude-pressure curtain animation.

The figure follows the sketch in ~/IMG_5966.jpg: a Robinson-projection column
mean map on the left and three longitude-pressure curtains on the right. Rows
show GEOS-Chem and AtmosTransport for the same species and timestamp.
"""

from __future__ import annotations

import argparse
import re
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
from netCDF4 import Dataset
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.spatial import cKDTree


warnings.filterwarnings("ignore", message="There are no gridspecs with layoutgrids.*")

G_ACCEL = 9.80665
M_DRY_AIR_KG_MOL = 28.96546e-3
RUN_START = datetime(2021, 12, 1)
GC_VAR = {
    "co2_natural": "SpeciesConcVV_CO2",
    "co2_fossil": "SpeciesConcVV_FossilCO2",
    "sf6": "SpeciesConcVV_SF6",
    "rn222": "SpeciesConcVV_Rn222",
}
GC_FLUX_VAR = {
    "co2_natural": "EmisCO2_Total",
    "co2_fossil": "Emis_FossilCO2_Total",
    "sf6": "EmisSF6",
    "rn222": "EmisRn_Soil",
}
MOLAR_MASS_KG_MOL = {
    "co2_natural": 44.0095e-3,
    "co2_fossil": 44.0095e-3,
    "sf6": 146.055e-3,
    "rn222": 222.0e-3,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--at", default=str(Path.home() / "data/AtmosTransport/output/catrine_geosit_c180_v4_fullphys_gchp_dec2021_smoke3d.nc"))
    p.add_argument("--gc", default=str(Path.home() / "data/AtmosTransport/catrine-geoschem-runs"))
    p.add_argument("--species", default="co2_fossil", choices=sorted(GC_VAR))
    p.add_argument("--out-dir", default=str(Path.home() / "data/AtmosTransport/output/catrine_geosit_c180_v4_fullphys_gchp_dec2021_smoke3d_animation"))
    p.add_argument("--fps", type=int, default=3)
    p.add_argument("--max-frames", type=int, default=0, help="0 means all matched frames")
    p.add_argument("--map-vmax", type=float, default=8.0)
    p.add_argument("--curtain-vmax", type=float, default=40.0)
    p.add_argument("--latitudes", default="40,0,-40")
    p.add_argument("--dlon", type=float, default=2.0)
    p.add_argument("--dp", type=float, default=10.0)
    p.add_argument("--at-log", default="", help="Run log used to read AT surface source rates.")
    return p.parse_args()


def lat_label(lat: float) -> str:
    if abs(lat) < 1e-9:
        return "Eq"
    return f"{abs(lat):g}{'N' if lat > 0 else 'S'}"


def add_section_guides(ax, section_lats: list[float]) -> None:
    lon = np.linspace(-180.0, 180.0, 721)
    for lat in section_lats:
        ax.plot(
            lon,
            np.full_like(lon, lat),
            transform=ccrs.PlateCarree(),
            color="0.25",
            linewidth=0.65,
            linestyle=(0, (1.2, 2.0)),
            alpha=0.65,
            zorder=5,
        )


def gc_datetime(path: Path) -> datetime | None:
    m = re.search(r"(\d{8})_(\d{4})z\.nc4$", path.name)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M")


def matched_times(at_path: Path, gc_dir: Path, max_frames: int) -> list[tuple[int, datetime, Path]]:
    gc_map = {dt: p for p in sorted(gc_dir.glob("GEOSChem.CATRINE_inst.*.nc4")) if (dt := gc_datetime(p))}
    out: list[tuple[int, datetime, Path]] = []
    with Dataset(at_path) as ds:
        hours = np.asarray(ds.variables["time"][:], dtype=float)
    for i, h in enumerate(hours):
        dt = RUN_START + timedelta(hours=float(h))
        gc_path = gc_map.get(dt)
        if gc_path is not None:
            out.append((i, dt, gc_path))
    return out if max_frames <= 0 else out[:max_frames]


def unit_xyz(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    lon = np.deg2rad(lon_deg.ravel())
    lat = np.deg2rad(lat_deg.ravel())
    return np.column_stack((np.cos(lat) * np.cos(lon),
                            np.cos(lat) * np.sin(lon),
                            np.sin(lat)))


def build_section_indices(lons: np.ndarray, lats: np.ndarray, section_lats: list[float], dlon: float):
    lon_samples = np.arange(-180.0, 180.0 + 0.5 * dlon, dlon)
    tree = cKDTree(unit_xyz(lons, lats))
    sections = []
    for lat in section_lats:
        xyz = unit_xyz(lon_samples, np.full_like(lon_samples, lat))
        _, flat = tree.query(xyz, k=1)
        idx = np.unravel_index(flat, lons.shape)
        sections.append((lat, lon_samples, idx))
    return sections


def infer_at_pressure_hpa(air_mass: np.ndarray, area: np.ndarray) -> np.ndarray:
    dp_pa = air_mass / area[:, :, :, None] * G_ACCEL
    p_edge = np.cumsum(dp_pa, axis=3)
    return (p_edge - 0.5 * dp_pa) / 100.0


def _read_cs_3d(ds: Dataset, name: str) -> np.ndarray:
    # Python netCDF4 exposes Julia-written variables as (nf, Y, X). The plotting
    # code uses the repo's Julia convention: (X, Y, nf).
    return np.asarray(ds.variables[name][:], dtype=np.float64).transpose(2, 1, 0)


def _read_cs_corners(ds: Dataset, name: str) -> np.ndarray:
    return np.asarray(ds.variables[name][:], dtype=np.float64).transpose(2, 1, 0)


def _read_cs_4d_at(ds: Dataset, name: str, time_index: int) -> np.ndarray:
    # netCDF4 order: (time, lev, nf, Y, X) -> (X, Y, nf, lev)
    return np.asarray(ds.variables[name][time_index, :, :, :, :], dtype=np.float64).transpose(3, 2, 1, 0)


def _read_gc_4d(ds: Dataset, name: str) -> np.ndarray:
    # netCDF4 order: (time, lev, nf, Y, X) -> (X, Y, nf, lev)
    return np.asarray(ds.variables[name][0, :, :, :, :], dtype=np.float64).transpose(3, 2, 1, 0)


def common_fields_at(at_ds: Dataset, gc_ds: Dataset, species: str, at_t: int):
    at_vmr = _read_cs_4d_at(at_ds, species, at_t) * 1e6
    at_air = _read_cs_4d_at(at_ds, "air_mass", at_t)
    area = _read_cs_3d(at_ds, "cell_area")
    at_p = infer_at_pressure_hpa(at_air, area)

    gc_name = GC_VAR[species]
    gc_vmr_surf_top = _read_gc_4d(gc_ds, gc_name) * 1e6
    gc_air_surf_top = _read_gc_4d(gc_ds, "Met_AD")
    gc_p_surf_top = _read_gc_4d(gc_ds, "Met_PMIDDRY")

    nz = min(at_vmr.shape[3], gc_vmr_surf_top.shape[3])
    at_vmr = at_vmr[:, :, :, -nz:]
    at_air = at_air[:, :, :, -nz:]
    at_p = at_p[:, :, :, -nz:]
    gc_vmr = gc_vmr_surf_top[:, :, :, :nz][:, :, :, ::-1]
    gc_air = gc_air_surf_top[:, :, :, :nz][:, :, :, ::-1]
    gc_p = gc_p_surf_top[:, :, :, :nz][:, :, :, ::-1]
    return at_vmr, at_air, at_p, gc_vmr, gc_air, gc_p


def at_log_path(at_path: Path, explicit: str) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    return Path.home() / "data/AtmosTransport/output/logs" / f"{at_path.stem}.log"


def read_at_flux_kg_s(path: Path, species: str) -> float | None:
    if not path.exists():
        return None
    pattern = re.compile(
        rf"Surface source {re.escape(species)} total model-storage rate:\s+"
        r"([0-9.eE+-]+)\s+kg_air_equiv/s"
    )
    text = path.read_text(errors="replace")
    match = pattern.search(text)
    if match is None:
        return None
    storage_rate = float(match.group(1))
    return storage_rate * MOLAR_MASS_KG_MOL[species] / M_DRY_AIR_KG_MOL


def gc_flux_kg_s(gc_ds: Dataset, species: str) -> float:
    flux = np.asarray(gc_ds.variables[GC_FLUX_VAR[species]][0, :, :, :], dtype=np.float64)
    area = np.asarray(gc_ds.variables["Met_AREAM2"][0, :, :, :], dtype=np.float64)
    return float(np.nansum(flux * area))


def column_mean_ppm(vmr_ppm: np.ndarray, air_mass: np.ndarray) -> np.ndarray:
    den = np.sum(air_mass, axis=3)
    out = np.sum(vmr_ppm * air_mass, axis=3) / den
    return np.where(np.isfinite(out), out, np.nan)


def global_burden_kg(vmr_ppm: np.ndarray, air_mass: np.ndarray, species: str) -> float:
    vmr = vmr_ppm * 1.0e-6
    factor = MOLAR_MASS_KG_MOL[species] / M_DRY_AIR_KG_MOL
    return float(np.nansum(vmr * air_mass * factor))


def burden_label(value_kg: float) -> str:
    return f"{value_kg:.4e} kg"


def budget_label(storage_kg: float, flux_kg_s: float | None, elapsed_s: float) -> str:
    if flux_kg_s is None:
        return f"Global storage: {burden_label(storage_kg)}\nGlobal Sum(flux): n/a"
    flux_integral = flux_kg_s * elapsed_s
    return (
        f"Global storage: {burden_label(storage_kg)}\n"
        f"Global Sum(flux) dt: {burden_label(flux_integral)}  ({flux_kg_s:.4e} kg/s)"
    )


def curtain(vmr_ppm: np.ndarray, p_hpa: np.ndarray, section, p_grid: np.ndarray) -> np.ndarray:
    _, lon_samples, idx = section
    out = np.full((len(p_grid), len(lon_samples)), np.nan)
    for n in range(len(lon_samples)):
        i, j, f = idx[0][n], idx[1][n], idx[2][n]
        p = p_hpa[i, j, f, :]
        v = vmr_ppm[i, j, f, :]
        ok = np.isfinite(p) & np.isfinite(v)
        if np.count_nonzero(ok) < 2:
            continue
        order = np.argsort(p[ok])
        out[:, n] = np.interp(p_grid, p[ok][order], v[ok][order], left=np.nan, right=np.nan)
    return out


def wrap_lon_180(lon: np.ndarray) -> np.ndarray:
    return ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0


def seam_cell_mask(corner_lon_wrapped: np.ndarray) -> np.ndarray:
    c00 = corner_lon_wrapped[:-1, :-1]
    c10 = corner_lon_wrapped[1:, :-1]
    c01 = corner_lon_wrapped[:-1, 1:]
    c11 = corner_lon_wrapped[1:, 1:]
    return (np.maximum.reduce([c00, c10, c01, c11]) -
            np.minimum.reduce([c00, c10, c01, c11])) > 180.0


def masked_panel_field(corner_lon: np.ndarray, field: np.ndarray) -> tuple[np.ndarray, np.ma.MaskedArray]:
    x = wrap_lon_180(corner_lon)
    z = np.maximum(np.asarray(field, dtype=float), 0.0)
    return x, np.ma.array(z, mask=seam_cell_mask(x))


def panel_mesh(ax, corner_lons, corner_lats, field, norm, cmap):
    artists = []
    transform = ccrs.PlateCarree()
    for f in range(6):
        x, z = masked_panel_field(corner_lons[:, :, f], field[:, :, f])
        y = np.asarray(corner_lats[:, :, f], dtype=float)
        artists.append(ax.pcolormesh(x, y, z, transform=transform, shading="flat",
                                     cmap=cmap, norm=norm, rasterized=True))
    return artists


def update_panel_mesh(artists, corner_lons, field):
    for f, artist in enumerate(artists):
        _x, z = masked_panel_field(corner_lons[:, :, f], field[:, :, f])
        artist.set_array(np.ma.ravel(z))


def main() -> None:
    args = parse_args()
    at_path = Path(args.at).expanduser()
    gc_dir = Path(args.gc).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    section_lats = [float(x) for x in args.latitudes.split(",")]

    pairs = matched_times(at_path, gc_dir, args.max_frames)
    if not pairs:
        raise SystemExit("No AT/GEOS-Chem timestamp matches found.")

    with Dataset(at_path) as at_ds:
        lons = _read_cs_3d(at_ds, "lons")
        lats = _read_cs_3d(at_ds, "lats")
        corner_lons = _read_cs_corners(at_ds, "corner_lons")
        corner_lats = _read_cs_corners(at_ds, "corner_lats")

    sections = build_section_indices(lons, lats, section_lats, args.dlon)
    p_grid = np.arange(0.0, 1000.0 + args.dp, args.dp)
    lon_edges = np.arange(-180.0 - 0.5 * args.dlon, 180.0 + args.dlon, args.dlon)
    p_edges = np.arange(-0.5 * args.dp, 1000.0 + args.dp, args.dp)

    cmap = LinearSegmentedColormap.from_list(
        "white_to_red", ["#fffdf7", "#fee8a8", "#fca85d", "#e34a33", "#7f0000"])
    map_norm = SymLogNorm(linthresh=0.05, linscale=0.35, vmin=0.0, vmax=args.map_vmax, base=10)
    curtain_norm = SymLogNorm(linthresh=0.05, linscale=0.35, vmin=0.0, vmax=args.curtain_vmax, base=10)
    at_flux_rate = read_at_flux_kg_s(at_log_path(at_path, args.at_log), args.species)

    fig = plt.figure(figsize=(15.8, 7.8), constrained_layout=False)
    map_axes = [
        fig.add_axes([0.018, 0.555, 0.545, 0.365], projection=ccrs.Robinson()),
        fig.add_axes([0.018, 0.105, 0.545, 0.365], projection=ccrs.Robinson()),
    ]
    cax_map = fig.add_axes([0.575, 0.145, 0.013, 0.705])

    curtain_left = 0.620
    curtain_width = 0.325
    curtain_height = 0.108
    top_ys = [0.720, 0.602, 0.484]
    bottom_ys = [0.312, 0.194, 0.076]
    curtain_axes = [
        [fig.add_axes([curtain_left, y, curtain_width, curtain_height]) for y in top_ys],
        [fig.add_axes([curtain_left, y, curtain_width, curtain_height]) for y in bottom_ys],
    ]
    cax_curtain = fig.add_axes([0.970, 0.090, 0.012, 0.780])
    fig.text(0.952, 0.480, "pressure [hPa]", rotation=90,
             ha="center", va="center", fontsize=9)

    for ax, title in zip(map_axes, ["GEOS-Chem column mean", "AtmosTransport column mean"]):
        ax.set_global()
        ax.set_title(title, fontsize=12)
        ax.add_feature(cfeature.LAND, facecolor="none", edgecolor="0.25", linewidth=0.4)
        ax.coastlines(linewidth=0.55, color="0.2")
        add_section_guides(ax, section_lats)

    with Dataset(at_path) as at_ds, Dataset(pairs[0][2]) as gc_ds:
        at_vmr, at_air, at_p, gc_vmr, gc_air, gc_p = common_fields_at(at_ds, gc_ds, args.species, pairs[0][0])
        gc_col = column_mean_ppm(gc_vmr, gc_air)
        at_col = column_mean_ppm(at_vmr, at_air)
        gc_burden = global_burden_kg(gc_vmr, gc_air, args.species)
        at_burden = global_burden_kg(at_vmr, at_air, args.species)
        gc_flux_rate = gc_flux_kg_s(gc_ds, args.species)
        gc_curtains = [curtain(gc_vmr, gc_p, s, p_grid) for s in sections]
        at_curtains = [curtain(at_vmr, at_p, s, p_grid) for s in sections]

    gc_mesh = panel_mesh(map_axes[0], corner_lons, corner_lats, gc_col, map_norm, cmap)
    at_mesh = panel_mesh(map_axes[1], corner_lons, corner_lats, at_col, map_norm, cmap)
    elapsed0 = (pairs[0][1] - RUN_START).total_seconds()
    burden_texts = [
        map_axes[0].text(
            0.012, -0.095, budget_label(gc_burden, gc_flux_rate, elapsed0),
            transform=map_axes[0].transAxes, fontsize=9.0, ha="left", va="bottom",
            clip_on=False,
            bbox=dict(facecolor="white", edgecolor="0.75", boxstyle="round,pad=0.25", alpha=0.92),
        ),
        map_axes[1].text(
            0.012, -0.095, budget_label(at_burden, at_flux_rate, elapsed0),
            transform=map_axes[1].transAxes, fontsize=9.0, ha="left", va="bottom",
            clip_on=False,
            bbox=dict(facecolor="white", edgecolor="0.75", boxstyle="round,pad=0.25", alpha=0.92),
        ),
    ]
    curtain_images = []
    for row, data_row, label in ((0, gc_curtains, "GEOS-Chem"), (1, at_curtains, "AtmosTransport")):
        row_images = []
        for col, (ax, data, (lat, _, _)) in enumerate(zip(curtain_axes[row], data_row, sections)):
            im = ax.pcolormesh(lon_edges, p_edges, np.maximum(data, 0.0),
                               shading="auto", cmap=cmap, norm=curtain_norm)
            ax.set_ylim(1000, 0)
            ax.set_xlim(-180, 180)
            ax.text(0.008, 0.93, f"{label}  {lat_label(lat)}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8.8,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=1.2))
            ax.grid(color="0.85", linewidth=0.35)
            if not (row == 1 and col == 2):
                ax.tick_params(axis="x", labelbottom=False)
            else:
                ax.set_xlabel("longitude")
            ax.yaxis.tick_right()
            if row == 0 and col == 1:
                ax.tick_params(axis="y", labelleft=False, labelright=True, pad=2)
            else:
                ax.tick_params(axis="y", labelleft=False, labelright=False, pad=1)
            row_images.append(im)
        curtain_images.append(row_images)

    cb = fig.colorbar(gc_mesh[0], cax=cax_map)
    cb.ax.set_title(f"X{args.species}\n[ppm]", fontsize=8, pad=5)
    cb.set_ticks([0, 0.05, 0.1, 0.5, 1, 2, 4, args.map_vmax])
    cb.set_ticklabels(["0", "0.05", "0.1", "0.5", "1", "2", "4", f"{args.map_vmax:g}"])
    cb = fig.colorbar(curtain_images[0][0], cax=cax_curtain)
    cb.ax.set_title(f"{args.species}\n[ppm]", fontsize=8, pad=5)
    cb.ax.yaxis.set_ticks_position("right")
    cb.ax.yaxis.set_label_position("right")
    cb.set_ticks([0, 0.1, 0.5, 1, 5, 10, 20, args.curtain_vmax])
    cb.set_ticklabels(["0", "0.1", "0.5", "1", "5", "10", "20", f"{args.curtain_vmax:g}"])
    title = fig.suptitle("", fontsize=14, fontweight="bold")

    def draw(frame_idx: int):
        at_i, dt, gc_path = pairs[frame_idx]
        with Dataset(at_path) as at_ds, Dataset(gc_path) as gc_ds:
            at_vmr, at_air, at_p, gc_vmr, gc_air, gc_p = common_fields_at(at_ds, gc_ds, args.species, at_i)
            gc_col = column_mean_ppm(gc_vmr, gc_air)
            at_col = column_mean_ppm(at_vmr, at_air)
            gc_burden = global_burden_kg(gc_vmr, gc_air, args.species)
            at_burden = global_burden_kg(at_vmr, at_air, args.species)
            gc_flux_rate = gc_flux_kg_s(gc_ds, args.species)
            gc_curtains = [curtain(gc_vmr, gc_p, s, p_grid) for s in sections]
            at_curtains = [curtain(at_vmr, at_p, s, p_grid) for s in sections]

        update_panel_mesh(gc_mesh, corner_lons, gc_col)
        update_panel_mesh(at_mesh, corner_lons, at_col)
        elapsed_s = (dt - RUN_START).total_seconds()
        burden_texts[0].set_text(budget_label(gc_burden, gc_flux_rate, elapsed_s))
        burden_texts[1].set_text(budget_label(at_burden, at_flux_rate, elapsed_s))
        for ims, fields in ((curtain_images[0], gc_curtains), (curtain_images[1], at_curtains)):
            for im, field in zip(ims, fields):
                im.set_array(np.ma.ravel(np.maximum(field, 0.0)))
        title.set_text(f"{dt:%Y-%m-%d %H:%M} UTC  |  {args.species}, symlog concentration color")
        return [*gc_mesh, *at_mesh, *burden_texts, *curtain_images[0], *curtain_images[1], title]

    draw(0)
    stem = f"{args.species}_column_map_curtains_symlog_at_vs_geoschem_layout_v4"
    png = out_dir / f"{stem}_first_frame.png"
    gif = out_dir / f"{stem}.gif"
    fig.savefig(png, dpi=150)
    ani = animation.FuncAnimation(fig, draw, frames=len(pairs), interval=1000 / args.fps, blit=False)
    ani.save(gif, writer=animation.PillowWriter(fps=args.fps), dpi=120)
    print(f"Saved first frame: {png}")
    print(f"Saved animation: {gif}")


if __name__ == "__main__":
    main()
