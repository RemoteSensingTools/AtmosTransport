#!/usr/bin/env python3
"""
Visualize 24h ERA5 CO2 transport: surface, ~750 hPa, and column-mean.

Usage:
    python3 scripts/visualization/plot_era5_24h_co2.py [input.nc] [output_dir] [snapshot_hours]

Defaults:
    input:  /tmp/era5_f64_startCO2_viz.nc
    output: /tmp/era5_co2_viz/
    snapshot_hours: 0,6,12,18,24
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
try:
    from matplotlib.colors import TwoSlopeNorm
except ImportError:
    from matplotlib.colors import Normalize as _Norm
    class TwoSlopeNorm(_Norm):
        """Fallback for matplotlib < 3.2."""
        def __init__(self, vmin, vcenter, vmax):
            super().__init__(vmin=vmin, vmax=vmax)
import netCDF4 as nc
from pathlib import Path
from datetime import datetime, timedelta

def load_data(nc_path):
    ds = nc.Dataset(nc_path, 'r')
    lon = ds.variables['lon'][:]
    lat = ds.variables['lat'][:]
    time_var = ds.variables['time']
    try:
        times = nc.num2date(time_var[:], time_var.units,
                            only_use_cftime_datetimes=False,
                            only_use_python_datetimes=True)
    except TypeError:
        times = nc.num2date(time_var[:], time_var.units)
        times = [datetime(t.year, t.month, t.day, t.hour, t.minute, t.second)
                 if not isinstance(t, datetime) else t for t in times]

    # NetCDF shape: (time, lat, lon) → keep as-is
    co2_sfc = ds.variables['co2_surface'][:, :, :] * 1e6      # → ppm
    co2_750 = ds.variables['co2_750hPa'][:, :, :] * 1e6
    co2_col = ds.variables['co2_column_mean'][:, :, :] * 1e6

    ds.close()
    # Return shape: (time, lat, lon)
    return lon, lat, times, co2_sfc, co2_750, co2_col


def plot_snapshot_panel(fig, axes, lon, lat, sfc, mid, col, time_str,
                        sfc_norm, mid_norm, col_norm, cmap):
    """Plot one row of 3 panels (surface, 750 hPa, column mean)."""
    titles = ['Surface CO₂', '~750 hPa CO₂', 'Column-mean CO₂']
    data = [sfc, mid, col]
    norms = [sfc_norm, mid_norm, col_norm]

    for ax, d, title, norm in zip(axes, data, titles, norms):
        im = ax.pcolormesh(lon, lat, d, cmap=cmap, norm=norm,
                           rasterized=True)
        ax.set_title(f'{title}\n{time_str}', fontsize=10)
        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)
        ax.set_aspect('auto')
        cb = fig.colorbar(im, ax=ax, orientation='horizontal',
                          pad=0.08, aspect=30, shrink=0.9)
        cb.set_label('ppm', fontsize=8)
        cb.ax.tick_params(labelsize=7)

    axes[0].set_ylabel('Latitude', fontsize=9)
    for ax in axes:
        ax.set_xlabel('Longitude', fontsize=9)
        ax.tick_params(labelsize=7)


def make_snapshots(lon, lat, times, co2_sfc, co2_750, co2_col, out_dir,
                   snapshot_hours=None):
    """Generate snapshot images at specified hours (default: 0, 6, 12, 18, 24)."""
    if snapshot_hours is None:
        snapshot_hours = [0, 6, 12, 18, 24]

    cmap = 'RdYlBu_r'

    sfc_vmin, sfc_vmax = np.percentile(co2_sfc, [1, 99])
    mid_vmin, mid_vmax = np.percentile(co2_750, [1, 99])
    col_vmin, col_vmax = np.percentile(co2_col, [1, 99])

    sfc_norm = Normalize(vmin=sfc_vmin, vmax=sfc_vmax)
    mid_norm = Normalize(vmin=mid_vmin, vmax=mid_vmax)
    col_norm = Normalize(vmin=col_vmin, vmax=col_vmax)

    for hi in snapshot_hours:
        ti = min(hi, len(times) - 1)
        t = times[ti]
        time_str = t.strftime('%Y-%m-%d %H:%M UTC')

        fig, axes = plt.subplots(1, 3, figsize=(18, 4.5),
                                 constrained_layout=True)
        plot_snapshot_panel(fig, axes, lon, lat,
                           co2_sfc[ti, :, :], co2_750[ti, :, :],
                           co2_col[ti, :, :],
                           time_str, sfc_norm, mid_norm, col_norm, cmap)
        fig.suptitle(f'ERA5 CO₂ Transport — {time_str}', fontsize=13,
                     fontweight='bold', y=1.02)
        out_path = out_dir / f'co2_snapshot_t{ti:02d}.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight',
                    facecolor='white')
        plt.close(fig)
        print(f'  Saved {out_path}')


def parse_snapshot_hours(raw):
    """Parse a comma-separated hour list like ``0,5,9,13,17,21,24``."""
    if raw is None or raw.strip() == "":
        return None
    hours = []
    for item in raw.split(','):
        item = item.strip()
        if not item:
            continue
        hours.append(int(item))
    return hours


def make_evolution_grid(lon, lat, times, co2_sfc, co2_750, co2_col, out_dir):
    """4x3 grid: rows = hours 0/6/12/18, cols = surface/750hPa/column."""
    snap_indices = [0, 6, 12, 18]
    snap_indices = [min(i, len(times)-1) for i in snap_indices]
    nrows = len(snap_indices)

    cmap = 'RdYlBu_r'
    sfc_vmin, sfc_vmax = np.percentile(co2_sfc, [1, 99])
    mid_vmin, mid_vmax = np.percentile(co2_750, [1, 99])
    col_vmin, col_vmax = np.percentile(co2_col, [1, 99])

    norms = [Normalize(vmin=sfc_vmin, vmax=sfc_vmax),
             Normalize(vmin=mid_vmin, vmax=mid_vmax),
             Normalize(vmin=col_vmin, vmax=col_vmax)]

    fig, axs = plt.subplots(nrows, 3, figsize=(18, 4 * nrows),
                            constrained_layout=True)

    col_titles = ['Surface CO₂ (ppm)', '~750 hPa CO₂ (ppm)',
                  'Column-mean CO₂ (ppm)']
    datasets = [co2_sfc, co2_750, co2_col]

    for row, ti in enumerate(snap_indices):
        t = times[ti]
        time_str = t.strftime('%Y-%m-%d %H:%M')
        for col in range(3):
            ax = axs[row, col]
            d = datasets[col][ti, :, :]
            im = ax.pcolormesh(lon, lat, d, cmap=cmap, norm=norms[col],
                               rasterized=True)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=11, fontweight='bold')
            ax.text(0.01, 0.97, time_str, transform=ax.transAxes,
                    fontsize=8, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
            ax.set_xlim(0, 360)
            ax.set_ylim(-90, 90)
            ax.set_aspect('auto')
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel('Lat', fontsize=8)
            if row == nrows - 1:
                ax.set_xlabel('Lon', fontsize=8)

    for col in range(3):
        sm = plt.cm.ScalarMappable(norm=norms[col], cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=axs[:, col], orientation='horizontal',
                          pad=0.04, aspect=40, shrink=0.8)
        cb.set_label('ppm', fontsize=8)
        cb.ax.tick_params(labelsize=7)

    fig.suptitle('ERA5 CO₂ Transport — 24h Evolution (Dec 1, 2021)',
                 fontsize=14, fontweight='bold', y=1.01)
    out_path = out_dir / 'co2_evolution_grid.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved {out_path}')


def make_difference_plot(lon, lat, times, co2_sfc, co2_750, co2_col, out_dir):
    """Show change from t=0 to t=end for all three levels."""
    t0, tf = 0, len(times) - 1

    dsfc = co2_sfc[tf, :, :] - co2_sfc[t0, :, :]
    dmid = co2_750[tf, :, :] - co2_750[t0, :, :]
    dcol = co2_col[tf, :, :] - co2_col[t0, :, :]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), constrained_layout=True)
    titles = ['ΔSurface CO₂', 'Δ~750 hPa CO₂', 'ΔColumn-mean CO₂']
    diffs = [dsfc, dmid, dcol]
    cmap = 'RdBu_r'

    for ax, d, title in zip(axes, diffs, titles):
        vlim = max(abs(np.percentile(d, 1)), abs(np.percentile(d, 99)))
        if vlim < 1e-10:
            vlim = 1.0
        norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
        im = ax.pcolormesh(lon, lat, d, cmap=cmap, norm=norm,
                           rasterized=True)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)
        ax.set_aspect('auto')
        ax.tick_params(labelsize=7)
        cb = fig.colorbar(im, ax=ax, orientation='horizontal',
                          pad=0.08, aspect=30, shrink=0.9)
        cb.set_label('Δppm', fontsize=8)
        cb.ax.tick_params(labelsize=7)

    axes[0].set_ylabel('Latitude', fontsize=9)
    for ax in axes:
        ax.set_xlabel('Longitude', fontsize=9)

    t0_str = times[t0].strftime('%H:%M')
    tf_str = times[tf].strftime('%H:%M')
    fig.suptitle(
        f'CO₂ Change Over 24h  ({times[t0].strftime("%Y-%m-%d")}  '
        f'{t0_str} → {tf_str} UTC)',
        fontsize=13, fontweight='bold', y=1.02)
    out_path = out_dir / 'co2_24h_difference.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved {out_path}')


def make_zonal_mean_hovmoller(lon, lat, times, co2_sfc, co2_750, co2_col,
                              out_dir):
    """Hovmoller diagram: zonal-mean CO2 vs latitude and time."""
    nt = len(times)
    hours = np.array([(t - times[0]).total_seconds() / 3600.0 for t in times])

    # Average over lon (axis=2) → (time, lat)
    zm_sfc = np.mean(co2_sfc, axis=2)
    zm_750 = np.mean(co2_750, axis=2)
    zm_col = np.mean(co2_col, axis=2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    titles = ['Surface CO₂', '~750 hPa CO₂', 'Column-mean CO₂']
    data = [zm_sfc, zm_750, zm_col]
    cmap = 'RdYlBu_r'

    for ax, d, title in zip(axes, data, titles):
        vmin, vmax = np.percentile(d, [2, 98])
        # d is (time, lat), transpose to (lat, time) for pcolormesh(hours, lat, ...)
        im = ax.pcolormesh(hours, lat, d.T, cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           rasterized=True)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Hour (UTC)', fontsize=9)
        ax.tick_params(labelsize=7)
        cb = fig.colorbar(im, ax=ax, orientation='horizontal',
                          pad=0.1, aspect=30, shrink=0.9)
        cb.set_label('ppm', fontsize=8)
        cb.ax.tick_params(labelsize=7)

    axes[0].set_ylabel('Latitude', fontsize=9)
    fig.suptitle('Zonal-Mean CO₂ — Hovmöller (Dec 1, 2021)',
                 fontsize=13, fontweight='bold', y=1.02)
    out_path = out_dir / 'co2_hovmoller_zonal.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved {out_path}')


def main():
    nc_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/era5_f64_startCO2_viz.nc'
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else '/tmp/era5_co2_viz')
    snapshot_hours = parse_snapshot_hours(sys.argv[3] if len(sys.argv) > 3 else None)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading {nc_path} ...')
    lon, lat, times, co2_sfc, co2_750, co2_col = load_data(nc_path)
    print(f'  {len(times)} time steps, {len(lon)}×{len(lat)} grid')
    print(f'  Surface CO₂: {co2_sfc.min():.1f} – {co2_sfc.max():.1f} ppm')
    print(f'  750 hPa CO₂: {co2_750.min():.1f} – {co2_750.max():.1f} ppm')
    print(f'  Column  CO₂: {co2_col.min():.1f} – {co2_col.max():.1f} ppm')

    print('\nGenerating evolution grid ...')
    make_evolution_grid(lon, lat, times, co2_sfc, co2_750, co2_col, out_dir)

    print('Generating 24h difference plot ...')
    make_difference_plot(lon, lat, times, co2_sfc, co2_750, co2_col, out_dir)

    print('Generating Hovmöller diagram ...')
    make_zonal_mean_hovmoller(lon, lat, times, co2_sfc, co2_750, co2_col, out_dir)

    print('Generating individual snapshots ...')
    make_snapshots(lon, lat, times, co2_sfc, co2_750, co2_col, out_dir,
                   snapshot_hours=snapshot_hours)

    print(f'\nAll plots saved to {out_dir}/')


if __name__ == '__main__':
    main()
