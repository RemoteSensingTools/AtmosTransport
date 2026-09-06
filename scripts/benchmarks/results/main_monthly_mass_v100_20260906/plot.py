"""Plot the archived 31-day compensated tracer totals."""
from pathlib import Path
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

source = Path(__file__).with_name("daily_drift.csv")
with source.open() as stream:
    rows = list(csv.DictReader(stream))
days = [int(row["day"]) for row in rows]
plt.rcParams.update({"font.size": 11, "svg.fonttype": "none"})
figure, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6), constrained_layout=True)
for axis, key, label, scale, unit, color in zip(
    axes,
    ("float32", "float64"),
    ("Float32", "Float64"),
    (1e6, 1e16),
    ("ppm", "× 10⁻¹⁶"),
    ("#1C8B62", "#6554A4"),
):
    values = [float(row[key]) * scale for row in rows]
    axis.plot(days, values, marker="o", markersize=3, linewidth=2, color=color)
    axis.set(ylabel=f"Relative drift ({unit})", title=label, xlim=(0, 31), ylim=(0, None))
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
axes[-1].set(xlabel="Transport day", xticks=[0, 7, 14, 21, 28, 31])
figure.suptitle("31-day tracer mass conservation\n"
               "C90 L66 · maximum absolute daily drift across six tracers", fontsize=12)
output = Path(sys.argv[1]) if len(sys.argv) > 1 else source.with_suffix(".svg")
figure.savefig(output)
if output.suffix == ".svg":
    output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")
plt.close(figure)
