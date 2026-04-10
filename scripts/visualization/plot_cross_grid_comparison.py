#!/usr/bin/env python3
"""
Cross-grid comparison plots for the 24-hour validation.

Reads snapshot NetCDFs from LatLon, ReducedGaussian, and CubedSphere
runs, produces panel comparison plots of column-mean CO2.

Usage:
    python scripts/visualization/plot_cross_grid_comparison.py \
        --ll <latlon_snapshot.nc> \
        --rg <reduced_gaussian_snapshot.nc> \
        [--cs <cubed_sphere_snapshot.nc>] \
        --output <output_dir>

Produces:
    - Column-mean CO2 at each snapshot hour
    - Difference maps (grid_X - LatLon) at each hour
    - Time series of global mean, min, max per grid
    - Zonal mean cross-sections
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

try:
    import netCDF4 as nc
except ImportError:
    print("ERROR: netCDF4 required. Install with: pip install netCDF4")
    sys.exit(1)

try:
    import cartopy.crs as ccrs
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("WARNING: cartopy not available, using simple plots")


def load_snapshot(path, tracer="natural_co2"):
    """Load a v2 snapshot NetCDF and return column-mean mixing ratio."""
    ds = nc.Dataset(path, 'r')

    # Try different variable name patterns
    for vname in [tracer, f"{tracer}_column_mean", "CO2", "co2",
                  "column_mean_vmr", "vmr"]:
        if vname in ds.variables:
            data = ds.variables[vname][:]
            break
    else:
        print(f"Available variables: {list(ds.variables.keys())}")
        raise KeyError(f"Could not find tracer variable in {path}")

    lons = ds.variables.get('lon', ds.variables.get('longitude', None))
    lats = ds.variables.get('lat', ds.variables.get('latitude', None))

    result = {
        'data': np.ma.filled(data, np.nan),
        'lons': lons[:] if lons is not None else None,
        'lats': lats[:] if lats is not None else None,
        'path': path,
    }

    # Check for time dimension
    if 'time' in ds.dimensions or 'snapshot' in ds.dimensions:
        tvar = ds.variables.get('time', ds.variables.get('snapshot_hours', None))
        if tvar is not None:
            result['times'] = tvar[:]

    ds.close()
    return result


def plot_global_map(ax, data_2d, lons, lats, title, vmin=None, vmax=None,
                    cmap='RdYlBu_r', diverging=False):
    """Plot a global 2D field on the given axes."""
    if diverging:
        absmax = max(abs(np.nanmin(data_2d)), abs(np.nanmax(data_2d)))
        vmin, vmax = -absmax, absmax
        cmap = 'RdBu_r'

    if HAS_CARTOPY and lons is not None and lats is not None:
        # Use cartopy projection
        im = ax.pcolormesh(lons, lats, data_2d.T if data_2d.shape[0] == len(lons) else data_2d,
                           transform=ccrs.PlateCarree(),
                           vmin=vmin, vmax=vmax, cmap=cmap, shading='auto')
        ax.coastlines(linewidth=0.5)
        ax.set_global()
    elif lons is not None and lats is not None:
        im = ax.pcolormesh(lons, lats,
                           data_2d.T if data_2d.shape[0] == len(lons) else data_2d,
                           vmin=vmin, vmax=vmax, cmap=cmap, shading='auto')
    else:
        im = ax.imshow(data_2d, vmin=vmin, vmax=vmax, cmap=cmap,
                       aspect='auto', origin='lower')

    ax.set_title(title, fontsize=10)
    return im


def plot_comparison_panel(ll_data, rg_data, cs_data, hour_idx, output_dir):
    """Create a multi-panel comparison for one time step."""
    n_grids = 2 + (1 if cs_data is not None else 0)

    fig, axes = plt.subplots(n_grids, 2, figsize=(16, 4 * n_grids),
                              subplot_kw={'projection': ccrs.Robinson()} if HAS_CARTOPY else {})

    # Column 1: absolute values
    # Column 2: difference from LatLon

    datasets = [('LatLon', ll_data), ('ReducedGaussian', rg_data)]
    if cs_data is not None:
        datasets.append(('CubedSphere', cs_data))

    # Determine common color scale from LatLon
    ll_field = ll_data['data'][hour_idx] if ll_data['data'].ndim > 2 else ll_data['data']
    vmin_abs = np.nanpercentile(ll_field, 1)
    vmax_abs = np.nanpercentile(ll_field, 99)

    for row, (name, dset) in enumerate(datasets):
        field = dset['data'][hour_idx] if dset['data'].ndim > 2 else dset['data']
        lons = dset['lons']
        lats = dset['lats']

        # Absolute value
        ax = axes[row, 0] if n_grids > 1 else axes[0]
        im = plot_global_map(ax, field, lons, lats, f"{name}",
                             vmin=vmin_abs, vmax=vmax_abs)
        plt.colorbar(im, ax=ax, shrink=0.7, label='CO2 VMR [ppm]')

        # Difference from LatLon
        if name == 'LatLon':
            ax2 = axes[row, 1] if n_grids > 1 else axes[1]
            ax2.text(0.5, 0.5, 'Reference', transform=ax2.transAxes,
                     ha='center', va='center', fontsize=14, color='gray')
            ax2.set_title('Reference (LatLon)')
        else:
            ax2 = axes[row, 1] if n_grids > 1 else axes[1]
            # Need to regrid to LL for difference — skip if shapes don't match
            if field.shape == ll_field.shape:
                diff = field - ll_field
                im2 = plot_global_map(ax2, diff, lons, lats,
                                      f"{name} - LatLon", diverging=True)
                plt.colorbar(im2, ax=ax2, shrink=0.7, label='Δ CO2 VMR [ppm]')
            else:
                ax2.text(0.5, 0.5, f'Shape mismatch\n{field.shape} vs {ll_field.shape}',
                         transform=ax2.transAxes, ha='center', va='center',
                         fontsize=10, color='red')
                ax2.set_title(f"{name} - LatLon (cannot diff)")

    hour_str = f"t={ll_data.get('times', [0])[hour_idx]:.0f}h" if 'times' in ll_data else f"idx={hour_idx}"
    fig.suptitle(f"Cross-Grid CO2 Comparison — {hour_str}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    outpath = os.path.join(output_dir, f"cross_grid_comparison_{hour_str.replace('=','')}.png")
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_time_series(ll_data, rg_data, cs_data, output_dir):
    """Plot global-mean CO2 time series for all grids."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    datasets = [('LatLon', ll_data, 'C0'),
                ('ReducedGaussian', rg_data, 'C1')]
    if cs_data is not None:
        datasets.append(('CubedSphere', cs_data, 'C2'))

    for name, dset, color in datasets:
        data = dset['data']
        if data.ndim == 2:
            means = [np.nanmean(data)]
            stds = [np.nanstd(data)]
            times = [0]
        else:
            means = [np.nanmean(data[t]) for t in range(data.shape[0])]
            stds = [np.nanstd(data[t]) for t in range(data.shape[0])]
            times = dset.get('times', range(data.shape[0]))

        axes[0].plot(times, means, 'o-', color=color, label=name, markersize=4)
        axes[1].plot(times, stds, 'o-', color=color, label=name, markersize=4)

    axes[0].set_ylabel('Global mean CO2 [ppm]')
    axes[0].set_title('Global Mean Column CO2')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel('Std dev CO2 [ppm]')
    axes[1].set_xlabel('Time [hours]')
    axes[1].set_title('Spatial Standard Deviation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = os.path.join(output_dir, "cross_grid_time_series.png")
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")


def main():
    parser = argparse.ArgumentParser(description="Cross-grid CO2 comparison plots")
    parser.add_argument('--ll', required=True, help='LatLon snapshot NetCDF')
    parser.add_argument('--rg', required=True, help='ReducedGaussian snapshot NetCDF')
    parser.add_argument('--cs', default=None, help='CubedSphere snapshot NetCDF (optional)')
    parser.add_argument('--output', default='plots/', help='Output directory')
    parser.add_argument('--tracer', default='natural_co2', help='Tracer variable name')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading snapshots...")
    ll = load_snapshot(args.ll, args.tracer)
    rg = load_snapshot(args.rg, args.tracer)
    cs = load_snapshot(args.cs, args.tracer) if args.cs else None

    print(f"  LatLon: shape={ll['data'].shape}")
    print(f"  ReducedGaussian: shape={rg['data'].shape}")
    if cs:
        print(f"  CubedSphere: shape={cs['data'].shape}")

    # Plot panels for each time step
    n_times = ll['data'].shape[0] if ll['data'].ndim > 2 else 1
    print(f"\nGenerating {n_times} comparison panels...")
    for t in range(n_times):
        plot_comparison_panel(ll, rg, cs, t, args.output)

    # Time series
    print("\nGenerating time series...")
    plot_time_series(ll, rg, cs, args.output)

    print("\nDone.")


if __name__ == '__main__':
    main()
