#!/usr/bin/env python3
"""Quick surface CO2 heatmap visualization from NetCDF output.
Fast Python alternative to Julia CairoMakie/Plots (no compilation).

Usage:
    python3 scripts/diagnostics/quick_viz.py /tmp/era5_v4_test.nc ~/www/catrina/
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 quick_viz.py <input.nc> <output_dir>")
        sys.exit(1)

    nc_path = sys.argv[1]
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import netCDF4 as nc
    except ImportError:
        from scipy.io import netcdf_file
        print("Using scipy fallback (limited)")
        sys.exit(1)

    ds = nc.Dataset(nc_path, 'r')
    lons = ds.variables['lon'][:]
    lats = ds.variables['lat'][:]
    times = ds.variables['time']
    time_units = times.units if hasattr(times, 'units') else 'hours'

    # Find surface field
    for var in ['co2_surface', 'co2_sfc', 'surface']:
        if var in ds.variables:
            sfc_var = var
            break
    else:
        print(f"No surface variable found. Available: {list(ds.variables.keys())}")
        sys.exit(1)

    sfc = ds.variables[sfc_var]
    nt = sfc.shape[2] if len(sfc.shape) == 3 else sfc.shape[0]
    print(f"Plotting {nt} timesteps from {sfc_var}")

    # Determine color range from first and last timestep
    s0 = sfc[:, :, 0] * 1e6 if len(sfc.shape) == 3 else sfc[0] * 1e6
    valid = s0[~np.isnan(s0)]
    if len(valid) > 0:
        vmin = max(np.percentile(valid, 1), 380)
        vmax = min(np.percentile(valid, 99), 450)
    else:
        vmin, vmax = 390, 440

    for t in range(nt):
        if len(sfc.shape) == 3:
            data = sfc[:, :, t] * 1e6  # VMR → ppm
        else:
            data = sfc[t] * 1e6

        if np.all(np.isnan(data)):
            print(f"  t={t}: all NaN, skipping")
            continue

        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        im = ax.pcolormesh(lons, lats, data.T, cmap='RdYlBu_r',
                          vmin=vmin, vmax=vmax, shading='auto')
        plt.colorbar(im, ax=ax, label='CO$_2$ (ppm)', shrink=0.8)

        try:
            import netCDF4
            time_val = netCDF4.num2date(times[t], times.units)
            title = f'Surface CO$_2$ — {time_val}'
        except:
            title = f'Surface CO$_2$ — t={t}'

        ax.set_title(title)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        fname = out_dir / f'sfc_co2_{t:02d}.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {fname}")

    # Also make column mean if available
    for var in ['co2_column_mean', 'co2_col']:
        if var in ds.variables:
            col = ds.variables[var]
            for t in range(col.shape[2] if len(col.shape) == 3 else col.shape[0]):
                if len(col.shape) == 3:
                    data = col[:, :, t] * 1e6
                else:
                    data = col[t] * 1e6
                if np.all(np.isnan(data)):
                    continue
                fig, ax = plt.subplots(1, 1, figsize=(12, 5))
                im = ax.pcolormesh(lons, lats, data.T, cmap='RdYlBu_r',
                                  vmin=vmin, vmax=vmax, shading='auto')
                plt.colorbar(im, ax=ax, label='CO$_2$ (ppm)', shrink=0.8)
                ax.set_title(f'Column Mean CO$_2$ — t={t}')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                fname = out_dir / f'col_co2_{t:02d}.png'
                fig.savefig(fname, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  Saved {fname}")
            break

    ds.close()
    print(f"\nAll plots saved to {out_dir}")

if __name__ == '__main__':
    main()
