"""
EDA 04 — Grid-to-finish relationship.

Answers:
  Q9.  How strongly does starting grid position predict finishing position,
       and how has that changed over time?
  Q10. What are the base rates of winning and podiums from specific grid
       slots (pole, front row, top-5)?

Notes on grid=0:
  In the Kaggle/Ergast data, grid=0 means the car started from the pit lane
  (usually after missing qualifying or taking a grid penalty that pushed
  them to the back). These rows are excluded from the main grid↔finish
  analysis because 0 isn't a real starting slot.

Correlation metric:
  Grid and finish position are both ordinal (lower is better). We use
  Spearman rank correlation because it doesn't assume a linear relationship
  and is robust to the long tail at the back of the grid.

Outputs:
  stdout : per-decade correlation, win/podium base rates, modern-era
           grid→finish expectation table
  png    : reports/eda/04_grid_finish/grid_finish.png (4-panel)
  csv    : reports/eda/04_grid_finish/grid_finish_corr_by_decade.csv
           reports/eda/04_grid_finish/win_rates_by_decade.csv
           reports/eda/04_grid_finish/grid_finish_joint_modern.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import DATA_RAW

OUT_DIR = Path(__file__).resolve().parents[2] / "reports" / "eda" / "04_grid_finish"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
races   = pd.read_csv(DATA_RAW / "races.csv")
results = pd.read_csv(DATA_RAW / "results.csv")

res = results.merge(races[["raceId", "year"]], on="raceId")


def _is_classified(pt) -> bool:
    try:
        int(pt); return True
    except (ValueError, TypeError):
        return False


res["classified"] = res["positionText"].apply(_is_classified)
res["finish_pos"] = pd.to_numeric(res["positionText"], errors="coerce")  # NaN if DNF

# Exclude grid=0 (pit lane start) from grid↔finish analysis
res_ok = res[res["grid"] > 0].copy()


def era_label(y: int) -> str:
    if y < 1980: return "pre-1980"
    if y < 2000: return "1980–1999"
    if y < 2014: return "2000–2013"
    return "2014–present"


res_ok["era"] = res_ok["year"].apply(era_label)

# ---------------------------------------------------------------------------
# Q9. Grid vs finish correlation (Spearman rank)
# ---------------------------------------------------------------------------
def spearman_grid_finish(df: pd.DataFrame) -> float:
    d = df[df["classified"]].dropna(subset=["finish_pos"])
    if len(d) < 20:
        return np.nan
    return d[["grid", "finish_pos"]].corr(method="spearman").iloc[0, 1]


corr_by_decade = (
    res_ok.assign(decade=(res_ok["year"] // 10) * 10)
          .groupby("decade").apply(spearman_grid_finish)
          .round(3)
          .rename("spearman_r")
)

print("=" * 60)
print("Q9. Grid → finish rank correlation (classified finishers only)")
print("=" * 60)
print("Spearman r: 1.0 means grid order = finish order; 0 means no relationship.")
print()
print(corr_by_decade.to_string())
print()

# Mean / median finishing position given grid position (modern era), for anchoring
modern_cls = (
    res_ok[(res_ok["year"] >= 2014) & (res_ok["classified"])]
        .dropna(subset=["finish_pos"])
)
grid_to_finish_modern = (
    modern_cls.groupby("grid")["finish_pos"]
              .agg(["mean", "median", "count"])
              .round(2)
              .loc[1:20]
)
print("Modern era (2014+): expected finishing position by grid slot")
print("(classified finishers only — DNFs excluded, so ‘mean finish’ is conditional on finishing)")
print(grid_to_finish_modern.to_string())
print()

# ---------------------------------------------------------------------------
# Q10. Win / podium base rates from specific grid slots, by decade
# ---------------------------------------------------------------------------
records = []
for decade in sorted(res_ok["year"].apply(lambda y: (y // 10) * 10).unique()):
    sub = res_ok[(res_ok["year"] >= decade) & (res_ok["year"] < decade + 10)]
    if len(sub) == 0:
        continue
    records.append({
        "decade":            decade,
        "P(win|pole)":       sub[sub["grid"] == 1]["positionOrder"].eq(1).mean(),
        "P(win|row1)":       sub[sub["grid"] <= 2]["positionOrder"].eq(1).mean(),
        "P(win|top5)":       sub[sub["grid"] <= 5]["positionOrder"].eq(1).mean(),
        "P(podium|pole)":    sub[sub["grid"] == 1]["positionOrder"].le(3).mean(),
        "P(podium|top5)":    sub[sub["grid"] <= 5]["positionOrder"].le(3).mean(),
        "P(classified|pole)":   sub[sub["grid"] == 1]["classified"].mean(),
        "P(classified|back)":   sub[sub["grid"] >= 15]["classified"].mean(),
    })
win_rates = pd.DataFrame(records).set_index("decade").round(3)

print("=" * 60)
print("Q10. Win / podium base rates by starting grid, by decade")
print("=" * 60)
print(win_rates.to_string())
print()

# ---------------------------------------------------------------------------
# Plots — 4-panel
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Panel (0,0): Spearman correlation year-by-year
def spearman_one_year(d: pd.DataFrame) -> float:
    d = d[d["classified"]].dropna(subset=["finish_pos"])
    if len(d) < 20: return np.nan
    return d[["grid", "finish_pos"]].corr(method="spearman").iloc[0, 1]

corr_by_year = res_ok.groupby("year").apply(spearman_one_year)
axes[0, 0].plot(corr_by_year.index, corr_by_year.values, marker=".", linewidth=1)
axes[0, 0].set_ylabel("Spearman r (grid, finish)")
axes[0, 0].set_xlabel("Year")
axes[0, 0].set_title("Grid → finish rank correlation, by year")
axes[0, 0].set_ylim(0, 1.0)
axes[0, 0].grid(alpha=0.3)

# Panel (0,1): P(win) given grid position, by era
era_order = ["pre-1980", "1980–1999", "2000–2013", "2014–present"]
for e in era_order:
    d = res_ok[res_ok["era"] == e]
    by_grid = d.groupby("grid")["positionOrder"].apply(lambda x: x.eq(1).mean())
    by_grid = by_grid.loc[1:20]
    axes[0, 1].plot(by_grid.index, by_grid.values, marker=".", linewidth=1, label=e)
axes[0, 1].set_xlabel("Grid position")
axes[0, 1].set_ylabel("P(win)")
axes[0, 1].set_title("Win probability given grid position, by era")
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.3)

# Panel (1,0): mean finishing position given grid, by era (classified only)
for e in era_order:
    d = res_ok[(res_ok["era"] == e) & (res_ok["classified"])].dropna(subset=["finish_pos"])
    mean_finish = d.groupby("grid")["finish_pos"].mean().loc[1:20]
    axes[1, 0].plot(mean_finish.index, mean_finish.values, marker=".", linewidth=1, label=e)
axes[1, 0].plot([1, 20], [1, 20], color="gray", linestyle="--", alpha=0.5, label="grid = finish")
axes[1, 0].set_xlabel("Grid position")
axes[1, 0].set_ylabel("Mean finish (classified)")
axes[1, 0].set_title("Mean finishing position given grid, by era")
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(alpha=0.3)

# Panel (1,1): heatmap of P(finish | grid) in modern era
joint = pd.crosstab(
    modern_cls["grid"].astype(int),
    modern_cls["finish_pos"].astype(int),
    normalize="index",
).loc[1:20, 1:20]
im = axes[1, 1].imshow(
    joint.values, cmap="viridis", aspect="auto", origin="upper",
    extent=[0.5, joint.shape[1] + 0.5, joint.shape[0] + 0.5, 0.5],
)
axes[1, 1].set_xlabel("Finish position")
axes[1, 1].set_ylabel("Grid position")
axes[1, 1].set_title("P(finish | grid), 2014+ classified finishers")
axes[1, 1].set_xticks(range(1, 21, 2))
axes[1, 1].set_yticks(range(1, 21, 2))
fig.colorbar(im, ax=axes[1, 1], label="Probability")

fig.tight_layout()
fig_path = OUT_DIR / "grid_finish.png"
fig.savefig(fig_path, dpi=120)
plt.close(fig)

# CSVs
corr_by_decade.to_csv(OUT_DIR / "grid_finish_corr_by_decade.csv")
win_rates.to_csv(OUT_DIR / "win_rates_by_decade.csv")
joint.to_csv(OUT_DIR / "grid_finish_joint_modern.csv")

print(f"Plot : {fig_path}")
print(f"Table: {OUT_DIR / 'grid_finish_corr_by_decade.csv'}")
print(f"Rates: {OUT_DIR / 'win_rates_by_decade.csv'}")
print(f"Joint: {OUT_DIR / 'grid_finish_joint_modern.csv'}")
