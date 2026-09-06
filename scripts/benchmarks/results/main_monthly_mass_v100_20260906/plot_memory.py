"""Plot process RSS and sampled device use, including startup and compilation."""
from pathlib import Path
from datetime import datetime
import csv
import json
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = Path(__file__).parent
plt.rcParams.update({"font.size": 11, "svg.fonttype": "none"})
figure, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6), constrained_layout=True)
summary = {}
for precision, color in (("Float32", "#1C8B62"), ("Float64", "#6554A4")):
    folder = here / precision
    with (folder / "host.tsv").open() as stream:
        recorded_host = list(csv.DictReader(stream, delimiter="\t"))
    # ps can observe the exited child with zero RSS/VSZ before GNU time reaps
    # it. Keep that row in the raw archive, but exclude it from live RSS plots.
    host = [row for row in recorded_host if int(row["rss_kib"]) > 0]
    host_time = [float(row["unix_seconds"]) for row in host]
    rss = [int(row["rss_kib"]) / 1024**2 for row in host]
    elapsed = [(t - host_time[0]) / 60 for t in host_time]
    axes[0].plot(elapsed, rss, color=color, linewidth=1, label=precision)

    with (folder / "device.csv").open() as stream:
        device = list(csv.DictReader(stream, skipinitialspace=True))
    device_time = [datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f") for row in device]
    used = [int(row["memory.used [MiB]"].split()[0]) / 1024 for row in device]
    device_elapsed = [(t - device_time[0]).total_seconds() / 60 for t in device_time]
    axes[1].plot(device_elapsed, used, color=color, linewidth=1, label=precision)

    resources = (folder / "resources.txt").read_text()
    maximum_rss = int(re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", resources)[1])
    row = {
        "process_peak_rss_gib": maximum_rss / 1024**2,
        "sampled_peak_rss_gib": max(rss),
        "sampled_peak_device_mib": max(used) * 1024,
        "host_samples": len(host),
        "discarded_zero_rss_samples": len(recorded_host) - len(host),
        "device_samples": len(device),
        "host_observed_seconds": host_time[-1] - host_time[0],
    }
    # Describe late-run variation without equating an allocator plateau with
    # a general absence of leaks. Startup and compilation remain in the plots.
    late = [value for t, value in zip(elapsed, rss) if t >= elapsed[-1] / 2]
    row["second_half_rss_range_gib"] = [min(late), max(late)]
    summary[precision] = row

for axis, label in zip(axes, ("Process RSS (GiB)", "Device use (GiB)")):
    axis.set(ylabel=label, ylim=(0, None))
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False)
axes[1].set(xlabel="Elapsed process time (minutes)")
figure.suptitle("Memory during 31-day transport runs\n"
               "C90 L66 · six tracers · includes compilation, mmap pages and CUDA pool", fontsize=12)
output = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "memory.svg"
figure.savefig(output)
if output.suffix == ".svg":
    output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")
plt.close(figure)
(here / "resources.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
