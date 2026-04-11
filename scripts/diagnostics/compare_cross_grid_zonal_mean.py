#!/usr/bin/env python3
"""
Cross-grid consistency check (Step 7 in the plan): compare LL vs RG
(or any two src_v2 snapshot NetCDFs) on the zonal mean of column-mean
CO2 over time.

Zonal binning is the simplest cross-grid test because it does not
require conservative regridding: we just bin each grid's cells by
latitude and average. LL and RG have slightly different latitude
row centers (LL: uniform, RG: Gauss-Legendre) so we interpolate both
onto a common latitude bin set for the comparison.

Usage:
    python3 scripts/diagnostics/compare_cross_grid_zonal_mean.py \
        <ll.nc> <rg.nc> [output.png]

Reports per-timestep stats (LL and RG zonal-mean min/max/mean,
LL↔RG RMSE and max |Δ|), and optionally writes a PNG with two
panels: left = zonal-mean col CO2 vs lat for each snapshot hour,
right = LL−RG difference vs lat.
"""
import sys
import numpy as np
import netCDF4 as nc


def load_zonal_mean_col(path):
    """Return (times, lat_centers, zonal_col_mean[time, lat]) for any
    supported grid.

    NCDatasets.jl writers use Julia column-major declaration order,
    which flips to row-major order on disk. Robust against any dim
    permutation.
    """
    ds = nc.Dataset(path, "r")
    try:
        grid_type = getattr(ds, "grid_type")
        times = np.asarray(ds.variables["time_hours"][:])
        col_var = ds.variables["co2_column_mean"]
        dims = col_var.dimensions
        data = np.asarray(col_var[:])

        if grid_type == "latlon":
            lat = np.asarray(ds.variables["lat"][:])
            # Permute so axes are ('time', 'lat', 'lon').
            axis_order = [dims.index("time"), dims.index("lat"), dims.index("lon")]
            data_tlln = np.transpose(data, axis_order)
            zonal = np.nanmean(data_tlln, axis=2)  # (time, lat)
        elif grid_type == "reduced_gaussian":
            lat_cell = np.asarray(ds.variables["lat_cell"][:])
            # Permute so axes are ('time', 'cell').
            axis_order = [dims.index("time"), dims.index("cell")]
            data_tc = np.transpose(data, axis_order)
            unique_lat = np.unique(lat_cell)
            zonal = np.empty((data_tc.shape[0], len(unique_lat)), dtype=float)
            for k, phi in enumerate(unique_lat):
                mask = lat_cell == phi
                zonal[:, k] = np.nanmean(data_tc[:, mask], axis=1)
            lat = unique_lat
        else:
            raise ValueError(f"Unsupported grid_type {grid_type!r}")

        return np.asarray(times), np.asarray(lat), zonal
    finally:
        ds.close()


def main():
    if len(sys.argv) < 3:
        print("Usage: compare_cross_grid_zonal_mean.py <a.nc> <b.nc> [out.png]")
        sys.exit(1)
    path_a, path_b = sys.argv[1], sys.argv[2]
    out_png = sys.argv[3] if len(sys.argv) > 3 else None

    ta, lat_a, zonal_a = load_zonal_mean_col(path_a)
    tb, lat_b, zonal_b = load_zonal_mean_col(path_b)

    # Check time alignment
    if not np.allclose(ta, tb):
        print(f"WARNING: time axes differ: A={ta}  B={tb}")

    # Interpolate both onto a common lat axis (the union of lat bins,
    # using linear interp). This is NOT conservative, but adequate for
    # a zonal-mean smoke-test where the signal is already averaged
    # longitudinally.
    lat_common = np.linspace(-90 + 1, 90 - 1, 60)  # 60 common lat bins

    def interp_to_common(lat_src, zonal_src):
        out = np.empty((zonal_src.shape[0], len(lat_common)))
        for i in range(zonal_src.shape[0]):
            out[i, :] = np.interp(lat_common, lat_src, zonal_src[i, :])
        return out

    za = interp_to_common(lat_a, zonal_a)
    zb = interp_to_common(lat_b, zonal_b)

    print(f"A: {path_a.split('/')[-1]}   lat {lat_a.min():.2f}..{lat_a.max():.2f}")
    print(f"B: {path_b.split('/')[-1]}   lat {lat_b.min():.2f}..{lat_b.max():.2f}")
    print()
    print(f"{'t(h)':>6}  {'A_min':>12}  {'A_max':>12}  {'B_min':>12}  {'B_max':>12}  "
          f"{'RMSE':>10}  {'max|dA-B|':>10}")
    for i, t in enumerate(ta):
        a = za[i, :]
        b = zb[i, :]
        diff = a - b
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        max_abs = float(np.max(np.abs(diff)))
        print(f"{float(t):6.2f}  {a.min():12.6e}  {a.max():12.6e}  "
              f"{b.min():12.6e}  {b.max():12.6e}  {rmse:10.3e}  {max_abs:10.3e}")

    if out_png is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
            colors = plt.cm.viridis(np.linspace(0, 1, len(ta)))
            for i, t in enumerate(ta):
                axs[0].plot(lat_common, za[i, :], color=colors[i], linestyle="-",
                            label=f"A t={float(t):.0f}h", linewidth=1.3)
                axs[0].plot(lat_common, zb[i, :], color=colors[i], linestyle="--",
                            linewidth=1.3)
                axs[1].plot(lat_common, za[i, :] - zb[i, :], color=colors[i],
                            label=f"t={float(t):.0f}h")
            axs[0].set_xlabel("Latitude (deg)")
            axs[0].set_ylabel("Zonal-mean column CO2 (VMR)")
            axs[0].set_title("Zonal-mean column CO2 (solid = A, dashed = B)")
            axs[0].legend(loc="best", fontsize=8, ncol=2)
            axs[0].grid(alpha=0.3)
            axs[1].set_xlabel("Latitude (deg)")
            axs[1].set_ylabel("A − B (VMR)")
            axs[1].set_title("Zonal-mean difference (LL − RG)")
            axs[1].axhline(0, color="k", linewidth=0.5)
            axs[1].legend(loc="best", fontsize=8)
            axs[1].grid(alpha=0.3)
            fig.suptitle(f"Cross-grid zonal-mean consistency\n"
                         f"A: {path_a.split('/')[-1]}\n"
                         f"B: {path_b.split('/')[-1]}",
                         fontsize=10)
            fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
            print(f"\nSaved {out_png}")
        except Exception as e:
            print(f"\nPlot error: {e}")


if __name__ == "__main__":
    main()
