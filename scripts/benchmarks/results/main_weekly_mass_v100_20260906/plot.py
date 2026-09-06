"""Plot the daily compensated-total comparisons exported by check_totals.jl."""
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
figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
for key, label, color in (
    ("seam_only_float32", "Earlier diffusion, Float32", "#355C8C"),
    ("conservative_float32", "Conservative diffusion, Float32", "#1C8B62"),
):
    values = [float(row[key]) * 1e6 for row in rows]
    axis.plot(days, values, marker="o", linewidth=2, color=color, label=label)
axis.set(xlabel="Transport day", ylabel="Maximum absolute relative mass drift (ppm)",
         title="Seven-day tracer mass conservation", xlim=(0, 7), ylim=(0, 4))
axis.set_xticks(days)
axis.grid(axis="y", alpha=0.2)
axis.spines[["top", "right"]].set_visible(False)
axis.legend(frameon=False, loc="upper left")
axis.text(0.98, 0.05, "C90 L66 · six tracers · daily compensated totals\n"
          "Float64 relative drift stays below 2 × 10⁻¹⁶", transform=axis.transAxes,
          ha="right", va="bottom", fontsize=9, color="#555555")
output = Path(sys.argv[1]) if len(sys.argv) > 1 else source.with_suffix(".svg")
figure.savefig(output)
plt.close(figure)
