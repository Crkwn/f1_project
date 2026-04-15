"""
EDA 06 — Circuit effects.

Answers:
  Q13. Do individual drivers over- or under-perform at specific circuits,
       beyond their own overall average finish? Measures 'circuit fit'.
  Q14. Does the grid-to-finish relationship differ across circuits? In
       particular, is grid order more preserved at street circuits (where
       overtaking is famously hard) than at high-speed permanent tracks?

Scope:
  Q13 runs on 2000–present to hold the driver/car pool roughly constant.
  Q14 runs on 2000–present and requires ≥10 races at the circuit in-window.

Caveats:
  - 'Circuit fit' residuals mix driver preference, car suitability for that
    circuit's characteristics, weather history at that circuit, and sheer
    luck. We'll surface the signal but not claim causal driver-only effect.
  - The street/permanent tag is hand-curated from a modest list of the
    most-raced circuits; the aim is directional comparison, not a full
    classification of every venue in the data.

Outputs:
  stdout : top-N over/under-performer residuals, per-circuit grid→finish r,
           street-vs-permanent comparison
  png    : reports/eda/06_circuits/circuits.png
  csv    : driver_circuit_residuals.csv, circuit_grid_finish_corr.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import DATA_RAW

OUT_DIR = Path(__file__).resolve().parents[2] / "reports" / "eda" / "06_circuits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
races    = pd.read_csv(DATA_RAW / "races.csv")
results  = pd.read_csv(DATA_RAW / "results.csv")
drivers  = pd.read_csv(DATA_RAW / "drivers.csv")
circuits = pd.read_csv(DATA_RAW / "circuits.csv")

res = results.merge(races[["raceId", "year", "circuitId"]], on="raceId")
res = res.merge(drivers[["driverId", "forename", "surname"]], on="driverId")
res["driver"] = res["forename"] + " " + res["surname"]
res = res.merge(circuits[["circuitId", "circuitRef", "name", "location", "country"]]
                .rename(columns={"name": "circuit_name"}), on="circuitId")

def _is_classified(pt) -> bool:
    try: int(pt); return True
    except: return False

res["classified"] = res["positionText"].apply(_is_classified)
res["finish_pos"] = pd.to_numeric(res["positionText"], errors="coerce")

modern = res[res["year"] >= 2000].copy()

# ---------------------------------------------------------------------------
# Q13. Driver × circuit residuals
# ---------------------------------------------------------------------------
cls = modern[modern["classified"]].dropna(subset=["finish_pos"])

# Driver overall average finish across all circuits (2000+)
driver_overall = cls.groupby("driverId")["finish_pos"].agg(["mean", "count"])
driver_overall = driver_overall[driver_overall["count"] >= 30]   # enough career data
driver_overall = driver_overall.rename(columns={"mean": "overall_mean",
                                                "count": "overall_n"})

# Driver × circuit mean
dc = cls.groupby(["driverId", "circuitId"])["finish_pos"].agg(["mean", "count"]).reset_index()
dc = dc.rename(columns={"mean": "circuit_mean", "count": "circuit_n"})
dc = dc[dc["circuit_n"] >= 4]  # at least 4 classified races at this circuit

# Join with overall and compute residual
dc = dc.merge(driver_overall.reset_index(), on="driverId")
dc["residual"] = dc["circuit_mean"] - dc["overall_mean"]   # negative = over-performs there

# Attach names
dc = dc.merge(drivers[["driverId", "forename", "surname"]], on="driverId")
dc["driver"] = dc["forename"] + " " + dc["surname"]
dc = dc.merge(circuits[["circuitId", "name"]].rename(columns={"name": "circuit"}), on="circuitId")

print("=" * 60)
print("Q13. Driver × circuit residuals (2000+)")
print("=" * 60)
print(f"Cells with ≥4 classified races at a circuit AND ≥30 career races: {len(dc)}")
print()
print("Top 10 driver × circuit OVER-performances (negative residual = better):")
top = dc.sort_values("residual").head(10)
print(top[["driver", "circuit", "circuit_n", "overall_mean",
           "circuit_mean", "residual"]].to_string(index=False))
print()
print("Top 10 driver × circuit UNDER-performances (positive residual = worse):")
bottom = dc.sort_values("residual", ascending=False).head(10)
print(bottom[["driver", "circuit", "circuit_n", "overall_mean",
              "circuit_mean", "residual"]].to_string(index=False))
print()

# How big a phenomenon is this overall? Std of residuals vs std of per-race noise.
print(f"Std of driver × circuit residual values : {dc['residual'].std():.2f} positions")
print(f"Std of raw finish positions (2000+)     : {cls['finish_pos'].std():.2f} positions")
print("  Residual std is the 'circuit fit' spread; raw std includes that plus noise")
print("  plus driver and car. Ratio tells us roughly the slice circuit-fit might hold.")
print()

# ---------------------------------------------------------------------------
# Q14. Per-circuit grid-to-finish correlation, with street/permanent tag
# ---------------------------------------------------------------------------
# Manual tag for most-raced circuits (covers the regulars). Anything not in
# this dict gets left as 'unclassified' and excluded from the split table.
STREET = {
    "monaco", "baku", "marina_bay", "jeddah", "vegas", "miami",
    "valencia", "detroit", "phoenix", "adelaide",
}
PERMANENT = {
    "silverstone", "spa", "monza", "suzuka", "americas", "catalunya",
    "hungaroring", "zandvoort", "imola", "interlagos", "nurburgring",
    "hockenheimring", "sepang", "shanghai", "estoril", "istanbul",
    "ricard", "rodriguez", "yas_marina", "ricardo_tormo", "albert_park",
    "BAK", "bahrain", "red_bull_ring", "osterreichring", "magny_cours",
    "hermanos_rodriguez",
}

def street_tag(ref: str) -> str:
    ref = ref.lower()
    if ref in STREET:    return "street"
    if ref in PERMANENT: return "permanent"
    return "unclassified"

circuits["track_type"] = circuits["circuitRef"].apply(street_tag)

# Grid-finish correlation per circuit (2000+, classified finishers only, grid>0)
cls_grid = modern[(modern["classified"]) & (modern["grid"] > 0)].dropna(subset=["finish_pos"])
per_circuit = []
for cid, g in cls_grid.groupby("circuitId"):
    if len(g) < 50:    # min races' worth of finishers
        continue
    r = g[["grid", "finish_pos"]].corr(method="spearman").iloc[0, 1]
    per_circuit.append({
        "circuitId": cid,
        "circuit": circuits.loc[circuits["circuitId"] == cid, "name"].iloc[0],
        "circuitRef": circuits.loc[circuits["circuitId"] == cid, "circuitRef"].iloc[0],
        "n_finishes": int(len(g)),
        "n_races": int(g["raceId"].nunique()),
        "spearman": r,
    })
circuit_corr = pd.DataFrame(per_circuit)
circuit_corr["track_type"] = circuit_corr["circuitRef"].apply(street_tag)

print("=" * 60)
print("Q14. Grid → finish rank correlation by circuit (2000+)")
print("=" * 60)
print(f"Circuits with ≥50 classified finishes in window: {len(circuit_corr)}")
print()
print("Tightest grid-preservation (top 10, highest Spearman):")
print(circuit_corr.sort_values("spearman", ascending=False).head(10)
      [["circuit", "track_type", "n_races", "spearman"]].to_string(index=False))
print()
print("Loosest grid-preservation (top 10, lowest Spearman):")
print(circuit_corr.sort_values("spearman").head(10)
      [["circuit", "track_type", "n_races", "spearman"]].to_string(index=False))
print()

# Street vs permanent, when tagged
tagged = circuit_corr[circuit_corr["track_type"] != "unclassified"]
summary = tagged.groupby("track_type")["spearman"].agg(["count", "mean", "median"]).round(3)
print("Street vs permanent, mean and median Spearman:")
print(summary.to_string())
print()
print("Reminder: our 'street' / 'permanent' labels are hand-tagged for the")
print("most common circuits only. This is directional, not a full taxonomy.")
print()

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Panel 1: residuals histogram
axes[0].hist(dc["residual"], bins=40, color="tab:blue", alpha=0.8)
axes[0].axvline(0, color="gray", linestyle="--", linewidth=0.7)
axes[0].set_xlabel("driver × circuit mean finish − driver overall mean finish")
axes[0].set_ylabel("count of (driver, circuit) cells")
axes[0].set_title("Circuit-fit residuals, 2000–present\n(negative = over-performance at that circuit)")
axes[0].grid(alpha=0.3)

# Panel 2: per-circuit Spearman, colored by track type
tagged_plot = circuit_corr.sort_values("spearman")
color_map = {"street": "tab:red", "permanent": "tab:blue", "unclassified": "tab:gray"}
colors = tagged_plot["track_type"].map(color_map)
axes[1].barh(range(len(tagged_plot)), tagged_plot["spearman"],
              color=colors, alpha=0.85)
axes[1].set_yticks(range(len(tagged_plot)))
axes[1].set_yticklabels(tagged_plot["circuit"], fontsize=7)
axes[1].set_xlabel("Spearman r (grid, finish)")
axes[1].set_title("Grid → finish correlation per circuit, 2000+")
axes[1].grid(alpha=0.3, axis="x")
# Manual legend
from matplotlib.patches import Patch
axes[1].legend(handles=[Patch(color=c, label=t) for t, c in color_map.items()],
               fontsize=8, loc="lower right")

fig.tight_layout()
fig_path = OUT_DIR / "circuits.png"
fig.savefig(fig_path, dpi=120)
plt.close(fig)

dc.to_csv(OUT_DIR / "driver_circuit_residuals.csv", index=False)
circuit_corr.to_csv(OUT_DIR / "circuit_grid_finish_corr.csv", index=False)

print(f"Plot : {fig_path}")
print(f"Table: {OUT_DIR / 'driver_circuit_residuals.csv'}")
print(f"Table: {OUT_DIR / 'circuit_grid_finish_corr.csv'}")
