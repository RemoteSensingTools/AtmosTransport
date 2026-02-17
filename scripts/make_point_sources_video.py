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
    co2 = ds.variables["co2_ppm"][:]  # (lon, lat, lev, time)
    ds.close()

    nx, ny, nz, ntime = co2.shape
    # Column average over lev -> (lon, lat, time)
    co2_col = np.nanmean(co2, axis=2)
    # Anomaly from initial; then (time, lon, lat) for looping
    ref = co2_col[:, :, 0]
    anomaly = np.transpose(co2_col - ref[:, :, np.newaxis], (2, 0, 1))  # (time, lon, lat)

    vmin = float(np.nanpercentile(anomaly, 2))
    vmax = float(np.nanpercentile(anomaly, 98))
    if vmax - vmin < 0.1:
        vmin, vmax = -1.0, 1.0

    lon2d, lat2d = np.meshgrid(lon, lat, indexing="ij")

    for t in range(ntime):
        fig = plt.figure(figsize=(10, 6))
        if HAS_CARTOPY:
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.coastlines(resolution="110m")
            ax.add_feature(cfeature.BORDERS, linestyle=":")
            ax.set_global()
            cf = ax.pcolormesh(
                lon2d, lat2d, anomaly[t].T,
                transform=ccrs.PlateCarree(),
                cmap="YlOrRd", vmin=vmin, vmax=vmax,
            )
        else:
            ax = plt.gca()
            cf = ax.pcolormesh(lon2d, lat2d, anomaly[t].T, cmap="YlOrRd", vmin=vmin, vmax=vmax)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_aspect("equal")

        plt.colorbar(cf, ax=ax, label="CO2 anomaly (ppm)")
        day = time_hours[t] / 24.0
        ax.set_title(f"Column-mean CO2 anomaly — Day {day:.2f}")
        plt.tight_layout()
        plt.savefig(os.path.join(frames_dir, f"frame_{t:04d}.png"), dpi=100, bbox_inches="tight")
        plt.close()

    ds.close()

    # Encode video with ffmpeg if available
    import subprocess
    ffmpeg = "ffmpeg"
    try:
        subprocess.run(
            [
                ffmpeg, "-y", "-framerate", "4", "-i",
                os.path.join(frames_dir, "frame_%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                outpath,
            ],
            check=True,
            capture_output=True,
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
