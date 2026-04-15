"""
EDA 07 — Temporal structure of driver performance.

Answers:
  Q15. How fast does a driver's average finishing position change season to
       season? How does that speed compare to within-season noise? This
       bounds how aggressively the score should update.
  Q16. When a driver sits out, how close are they to their pre-gap level when
       they return? Informs whether the score should decay (widen uncertainty)
       during inactivity.

Scope:
  1950–present for the year-over-year analysis. Minimum 5 classified finishes
  per season included, to avoid one-race substitutes contaminating the means.

Caveats:
  - 'Change in avg finishing position' is a composite: it includes genuine
    ability drift, car changes, team changes, and rule-package changes.
    We're measuring the *envelope* of change, not the pure driver delta.
  - For the gaps analysis, 'returning after a gap' is a selected population
    (drivers who came back at all — usually the ones teams still wanted).
    This is a survivor-bias wary number.

Outputs:
  stdout : distributions and medians for year-on-year delta and within-season
           std; tables of drivers returning after gaps
  png    : reports/eda/07_temporal/temporal.png
  csv    : driver_year_means.csv, gap_returns.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import DATA_RAW

OUT_DIR = Path(__file__).resolve().parents[2] / "reports" / "eda" / "07_temporal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
races   = pd.read_csv(DATA_RAW / "races.csv")
results = pd.read_csv(DATA_RAW / "results.csv")
drivers = pd.read_csv(DATA_RAW / "drivers.csv")

res = results.merge(races[["raceId", "year"]], on="raceId")
res = res.merge(drivers[["driverId", "forename", "surname"]], on="driverId")
res["driver"] = res["forename"] + " " + res["surname"]

def _is_classified(pt) -> bool:
    try: int(pt); return True
    except: return False

res["classified"] = res["positionText"].apply(_is_classified)
res["finish_pos"] = pd.to_numeric(res["positionText"], errors="coerce")

cls = res[res["classified"]].dropna(subset=["finish_pos"])

# Driver × year: mean finish and within-season std
dy = cls.groupby(["driverId", "year"])["finish_pos"].agg(
    mean_finish="mean", std_finish="std", n_finishes="size"
).reset_index()
dy = dy[dy["n_finishes"] >= 5]
dy["driver"] = dy["driverId"].map(
    drivers.set_index("driverId").apply(lambda r: f"{r['forename']} {r['surname']}", axis=1)
)

# ---------------------------------------------------------------------------
# Q15. Year-over-year change and within-season volatility
# ---------------------------------------------------------------------------
# Sort by driver then year
dy_sorted = dy.sort_values(["driverId", "year"]).reset_index(drop=True)
# Compute shift per driver
dy_sorted["prev_mean"]  = dy_sorted.groupby("driverId")["mean_finish"].shift(1)
dy_sorted["prev_year"]  = dy_sorted.groupby("driverId")["year"].shift(1)
dy_sorted["gap"]        = dy_sorted["year"] - dy_sorted["prev_year"]
# Consecutive-year deltas only (gap == 1)
consec = dy_sorted[dy_sorted["gap"] == 1].copy()
consec["delta"] = consec["mean_finish"] - consec["prev_mean"]

print("=" * 60)
print("Q15. Year-over-year change in mean finishing position")
print("=" * 60)
print(f"Consecutive driver-year pairs (same driver, back-to-back seasons, each ≥5 finishes): {len(consec)}")
print()

# Distribution summary
print("Distribution of year-over-year |delta| in mean finish:")
print(f"  median : {consec['delta'].abs().median():.2f}")
print(f"  mean   : {consec['delta'].abs().mean():.2f}")
print(f"  p75    : {consec['delta'].abs().quantile(0.75):.2f}")
print(f"  p95    : {consec['delta'].abs().quantile(0.95):.2f}")
print()

# Within-season std
print("Distribution of within-season std of finish positions (≥5 finishes in season):")
print(f"  median : {dy['std_finish'].median():.2f}")
print(f"  mean   : {dy['std_finish'].mean():.2f}")
print(f"  p75    : {dy['std_finish'].quantile(0.75):.2f}")
print(f"  p95    : {dy['std_finish'].quantile(0.95):.2f}")
print()
print("Comparison:")
print(f"  Median |year-over-year delta| : {consec['delta'].abs().median():.2f}")
print(f"  Median within-season std       : {dy['std_finish'].median():.2f}")
print("  If within-season std >> y-o-y delta, noise dominates — season-level")
print("  'ability change' is hard to see against the week-to-week volatility.")
print()

# Splitting by era
def era_label(y):
    if y < 1980: return "pre-1980"
    if y < 2000: return "1980–1999"
    if y < 2014: return "2000–2013"
    return "2014–present"

consec["era"] = consec["year"].apply(era_label)
dy["era"] = dy["year"].apply(era_label)
print("Year-over-year |delta| and within-season std, by era:")
era_stats = pd.DataFrame({
    "yoy_abs_delta_median": consec.groupby("era")["delta"].apply(lambda s: s.abs().median()).round(2),
    "within_season_std_median": dy.groupby("era")["std_finish"].median().round(2),
    "n_pairs": consec.groupby("era").size(),
})
print(era_stats.to_string())
print()

# ---------------------------------------------------------------------------
# Q16. Drivers returning after a gap
# ---------------------------------------------------------------------------
# gap > 1 means driver sat out one or more seasons in between.
gap_rows = dy_sorted[(dy_sorted["gap"] > 1)].copy()
gap_rows["delta"] = gap_rows["mean_finish"] - gap_rows["prev_mean"]

print("=" * 60)
print("Q16. Drivers returning after one or more missed seasons")
print("=" * 60)
print(f"Return events (seasons with ≥5 finishes on either side): {len(gap_rows)}")
print()

if len(gap_rows):
    print("Headline: typical gap-return mean finish delta (positive = worse after return):")
    print(f"  median delta : {gap_rows['delta'].median():.2f}")
    print(f"  mean delta   : {gap_rows['delta'].mean():.2f}")
    print(f"  vs consec-year median delta : {consec['delta'].median():.2f}")
    print()
    # By gap length
    g = gap_rows.copy()
    g["gap_length"] = g["gap"].astype(int) - 1   # years actually missed
    print("By length of gap (years missed):")
    print(g.groupby("gap_length")["delta"].agg(["count", "mean", "median"]).round(2).to_string())
    print()

    # Some named examples
    print("Notable returns (largest absolute delta, top 10):")
    print(g.sort_values("delta", key=lambda s: s.abs(), ascending=False).head(10)
          [["driver", "prev_year", "year", "gap_length", "prev_mean",
            "mean_finish", "delta"]].to_string(index=False))

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Panel 1: yoy delta vs within-season std, by era
eras_in_order = ["pre-1980", "1980–1999", "2000–2013", "2014–present"]
yoy_data    = [consec[consec["era"] == e]["delta"].abs().dropna().values for e in eras_in_order]
within_data = [dy[dy["era"] == e]["std_finish"].dropna().values          for e in eras_in_order]
# Grouped boxplot, side by side per era
positions_a = [i - 0.18 for i in range(1, len(eras_in_order) + 1)]
positions_b = [i + 0.18 for i in range(1, len(eras_in_order) + 1)]

bp_a = axes[0].boxplot(yoy_data, positions=positions_a, widths=0.32,
                        showfliers=False, patch_artist=True)
bp_b = axes[0].boxplot(within_data, positions=positions_b, widths=0.32,
                        showfliers=False, patch_artist=True)
for patch in bp_a["boxes"]: patch.set_facecolor("tab:blue")
for patch in bp_b["boxes"]: patch.set_facecolor("tab:orange")

axes[0].set_xticks(range(1, len(eras_in_order) + 1))
axes[0].set_xticklabels(eras_in_order, fontsize=9)
axes[0].set_ylabel("Positions")
axes[0].set_title("Ability-change signal vs within-season noise, by era")
from matplotlib.patches import Patch
axes[0].legend(handles=[Patch(color="tab:blue", label="|year-over-year Δ mean finish|"),
                         Patch(color="tab:orange", label="within-season std of finish")],
                fontsize=8, loc="upper right")
axes[0].grid(alpha=0.3, axis="y")

# Panel 2: gap-return deltas (where data exists)
if len(gap_rows):
    g = gap_rows.copy()
    g["gap_length"] = g["gap"].astype(int) - 1
    axes[1].scatter(g["gap_length"], g["delta"], alpha=0.55, s=30)
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.7)
    axes[1].set_xlabel("Years missed (gap length)")
    axes[1].set_ylabel("Δ mean finish (after − before)")
    axes[1].set_title("Finish change after returning from a gap\n(positive = worse on return)")
    axes[1].grid(alpha=0.3)

fig.tight_layout()
fig_path = OUT_DIR / "temporal.png"
fig.savefig(fig_path, dpi=120)
plt.close(fig)

dy.to_csv(OUT_DIR / "driver_year_means.csv", index=False)
gap_rows.to_csv(OUT_DIR / "gap_returns.csv", index=False)

print(f"\nPlot : {fig_path}")
print(f"Table: {OUT_DIR / 'driver_year_means.csv'}")
print(f"Table: {OUT_DIR / 'gap_returns.csv'}")
