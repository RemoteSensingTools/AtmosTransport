#!/usr/bin/env python3
"""
Make a video from the point-source CO2 NetCDF output.

Reads ~/data/metDrivers/geosfp/point_sources_co2.nc (or path from argv),
plots column-averaged CO2 anomaly (ppm) for each time step, and writes
point_sources_co2.mp4 in the same directory.

Requires: numpy, netCDF4, matplotlib. For a map background: pip install cartopy

Usage:
  python scripts/make_point_sources_video.py [path/to/point_sources_co2.nc]
"""

import os
import sys
import numpy as np

try:
    import netCDF4 as nc
except ImportError:
    print("Install netCDF4: pip install netCDF4")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("Install matplotlib: pip install matplotlib")
    sys.exit(1)

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

# Default input/output
DEFAULT_NC = os.path.expanduser("~/data/metDrivers/geosfp/point_sources_co2.nc")


def main():
    ncpath = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_NC
    if not os.path.isfile(ncpath):
        print(f"File not found: {ncpath}")
        print("Run first: julia --project=. scripts/run_point_sources_geosfp.jl")
        sys.exit(1)

    outpath = ncpath.replace(".nc", ".mp4")
    outdir = os.path.dirname(ncpath)
    frames_dir = os.path.join(outdir, "video_frames")
    os.makedirs(frames_dir, exist_ok=True)

    ds = nc.Dataset(ncpath, "r")
    lon = ds.variables["lon"][:]
    lat = ds.variables["lat"][:]
    time_hours = ds.variables["time_hours"][:]
    co2 = ds.variables["co2_ppm"][:]
    dims = ds.variables["co2_ppm"].dimensions
    ds.close()

    nlon, nlat = len(lon), len(lat)
    axis_lev = dims.index("lev")
    axis_time = dims.index("time")
    # Column average over lev
    co2_col = np.nanmean(co2, axis=axis_lev)  # e.g. (time, lat, lon) or (lon, lat, time)
    # Move time to first axis
    co2_col = np.moveaxis(co2_col, axis_time, 0)
    ntime = co2_col.shape[0]
    # Anomaly from first time
    ref = co2_col[0]
    anomaly = co2_col - ref
    # Ensure (ntime, nlon, nlat) for pcolormesh(lon2d, lat2d, anomaly[t]); lon2d is (nlon, nlat)
    if anomaly.shape[1] == nlat and anomaly.shape[2] == nlon:
        anomaly = np.transpose(anomaly, (0, 2, 1))  # (time, lat, lon) -> (time, lon, lat)
    elif anomaly.shape[1] != nlon or anomaly.shape[2] != nlat:
        raise ValueError("co2_ppm dims after avg: expected (time, lon, lat) or (time, lat, lon)")

    # Use first 30 frames (or all if fewer) for color scale so blow-up doesn't dominate
    n_sensible = min(30, ntime)
    sensible = anomaly[:n_sensible]
    vmin = float(np.nanpercentile(sensible, 2))
    vmax = float(np.nanpercentile(sensible, 98))
    if vmax - vmin < 0.01:
        vmin, vmax = 0.0, 1.0
    # Clip displayed data to this range so later frames don't wash out the colorbar
    plot_data = np.clip(anomaly, vmin, vmax)

    lon2d, lat2d = np.meshgrid(lon, lat, indexing="ij")

    for t in range(ntime):
        fig = plt.figure(figsize=(10, 6))
        if HAS_CARTOPY:
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.coastlines(resolution="110m")
            ax.add_feature(cfeature.BORDERS, linestyle=":")
            ax.set_global()
            cf = ax.pcolormesh(
                lon2d, lat2d, plot_data[t],
                transform=ccrs.PlateCarree(),
                cmap="YlOrRd", vmin=vmin, vmax=vmax,
            )
        else:
            ax = plt.gca()
            cf = ax.pcolormesh(lon2d, lat2d, plot_data[t], cmap="YlOrRd", vmin=vmin, vmax=vmax)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_aspect("equal")

        plt.colorbar(cf, ax=ax, label="CO2 (ppm, column-mean)")
        day = time_hours[t] / 24.0
        ax.set_title("Column-mean CO2 — Day {:.2f}".format(day))
        plt.tight_layout()
        plt.savefig(os.path.join(frames_dir, f"frame_{t:04d}.png"), dpi=100, bbox_inches="tight")
        plt.close()

    # Encode video with ffmpeg if available
    import subprocess
    ffmpeg = "ffmpeg"
    try:
        subprocess.run(
            [
                ffmpeg, "-y", "-framerate", "4", "-i",
                os.path.join(frames_dir, "frame_%04d.png"),
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                outpath,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"Video written: {outpath}")
        # Optionally remove frames to save space
        for f in os.listdir(frames_dir):
            os.remove(os.path.join(frames_dir, f))
        os.rmdir(frames_dir)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"FFmpeg not found or failed: {e}")
        print(f"Frames saved in: {frames_dir}")
        print("Encode manually: ffmpeg -framerate 4 -i video_frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p out.mp4")

    return 0


if __name__ == "__main__":
    sys.exit(main())
