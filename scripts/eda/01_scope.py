"""
EDA 01 — Scope and shape of the F1 historical dataset.

Answers:
  Q1. How many seasons does the data cover, and how many races per season?
  Q2. How many drivers took part each season, and how big are the fields per race?
  Q3. How many constructors per season, and how concentrated are wins at the top?

Outputs:
  stdout : summary tables (decade-level rollups + a few headline numbers)
  png    : reports/eda/01_scope/scope_by_season.png
  csv    : reports/eda/01_scope/scope_by_decade.csv
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Make project imports work when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import DATA_RAW

OUT_DIR = Path(__file__).resolve().parents[2] / "reports" / "eda" / "01_scope"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
races   = pd.read_csv(DATA_RAW / "races.csv")
results = pd.read_csv(DATA_RAW / "results.csv")

# Attach year to every result row for easy grouping
res = results.merge(races[["raceId", "year"]], on="raceId")


def _by_decade(s: pd.Series) -> pd.Series:
    """Group a year-indexed series into decades (1950, 1960, 1970, ...)."""
    return s.groupby((s.index // 10) * 10)


# ---------------------------------------------------------------------------
# Q1. Races per season
# ---------------------------------------------------------------------------
races_per_year = races.groupby("year").size().rename("races")

print("=" * 60)
print("Q1. Races per season")
print("=" * 60)
print(f"Years covered     : {races_per_year.index.min()}–{races_per_year.index.max()}")
print(f"Seasons           : {len(races_per_year)}")
print(f"Total races       : {races_per_year.sum()}")
print(f"Fewest in a season: {races_per_year.min()} ({races_per_year.idxmin()})")
print(f"Most in a season  : {races_per_year.max()} ({races_per_year.idxmax()})")
print()

races_decadal = _by_decade(races_per_year).agg(["mean", "min", "max", "count"]).round(1)
races_decadal.columns = ["mean_races", "min_races", "max_races", "seasons"]
print("By decade:")
print(races_decadal.to_string())
print()

# ---------------------------------------------------------------------------
# Q2. Drivers per season and per race
# ---------------------------------------------------------------------------
drivers_per_year = res.groupby("year")["driverId"].nunique().rename("drivers_per_season")
drivers_per_race = (
    res.groupby(["year", "raceId"])["driverId"].nunique()
       .reset_index(name="field_size")
)
field_by_year = drivers_per_race.groupby("year")["field_size"].agg(["mean", "min", "max"]).round(1)

print("=" * 60)
print("Q2. Drivers per season and field size per race")
print("=" * 60)
print("Unique drivers per season — by decade:")
print(_by_decade(drivers_per_year).agg(["mean", "min", "max"]).round(1).to_string())
print()
print("Field size per race (mean/min/max within year, then averaged by decade):")
decade_field = pd.DataFrame({
    "mean_field": field_by_year["mean"].groupby((field_by_year.index // 10) * 10).mean().round(1),
    "min_field":  field_by_year["min"].groupby((field_by_year.index // 10) * 10).min(),
    "max_field":  field_by_year["max"].groupby((field_by_year.index // 10) * 10).max(),
})
print(decade_field.to_string())
print()

# ---------------------------------------------------------------------------
# Q3. Constructors per season and win concentration
# ---------------------------------------------------------------------------
constructors_per_year = res.groupby("year")["constructorId"].nunique().rename("constructors")

print("=" * 60)
print("Q3. Constructors per season + win concentration")
print("=" * 60)
print("Unique constructors per season — by decade:")
print(_by_decade(constructors_per_year).agg(["mean", "min", "max"]).round(1).to_string())
print()

# Win concentration: what share of a season's wins go to the top-3 constructors?
wins = res[res["positionOrder"] == 1]
wins_by_con_year = wins.groupby(["year", "constructorId"]).size().reset_index(name="wins")
top3 = (
    wins_by_con_year.sort_values(["year", "wins"], ascending=[True, False])
                    .groupby("year").head(3)
                    .groupby("year")["wins"].sum()
)
total_wins_year = wins_by_con_year.groupby("year")["wins"].sum()
top3_share = (top3 / total_wins_year).rename("top3_win_share")

# Winner-take-all: share of wins held by the best constructor that season
top1 = (
    wins_by_con_year.sort_values(["year", "wins"], ascending=[True, False])
                    .groupby("year").head(1)
                    .groupby("year")["wins"].sum()
)
top1_share = (top1 / total_wins_year).rename("top1_win_share")

print("Share of season wins going to top-1 / top-3 constructors — by decade:")
conc_decadal = pd.DataFrame({
    "top1_share": _by_decade(top1_share).mean().round(3),
    "top3_share": _by_decade(top3_share).mean().round(3),
})
print(conc_decadal.to_string())
print()

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

axes[0].plot(races_per_year.index, races_per_year.values, marker=".", linewidth=1)
axes[0].set_ylabel("Races per season")
axes[0].set_title("F1 scope by season, 1950–present")
axes[0].grid(alpha=0.3)

axes[1].plot(drivers_per_year.index, drivers_per_year.values,
             marker=".", linewidth=1, label="Unique drivers / season")
axes[1].plot(field_by_year.index, field_by_year["mean"],
             marker=".", linewidth=1, label="Mean field size / race")
axes[1].set_ylabel("Drivers")
axes[1].legend(loc="upper left")
axes[1].grid(alpha=0.3)

axes[2].plot(constructors_per_year.index, constructors_per_year.values,
             marker=".", linewidth=1, color="tab:green", label="Constructors / season")
axes[2].set_ylabel("Constructors")
ax2b = axes[2].twinx()
ax2b.plot(top1_share.index, top1_share.values,
          marker=".", linewidth=1, color="tab:red", alpha=0.7, label="Top-1 win share")
ax2b.plot(top3_share.index, top3_share.values,
          marker=".", linewidth=1, color="tab:orange", alpha=0.7, label="Top-3 win share")
ax2b.set_ylabel("Fraction of season wins")
ax2b.set_ylim(0, 1.05)
axes[2].set_xlabel("Year")
axes[2].grid(alpha=0.3)

# Combine legends on bottom panel
lines1, labels1 = axes[2].get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
axes[2].legend(lines1 + lines2, labels1 + labels2, loc="upper left")

fig.tight_layout()
fig_path = OUT_DIR / "scope_by_season.png"
fig.savefig(fig_path, dpi=120)
plt.close(fig)

# Save a decade-level summary CSV for quick re-reference
decade_summary = pd.DataFrame({
    "mean_races_per_season":        races_decadal["mean_races"],
    "mean_drivers_per_season":      _by_decade(drivers_per_year).mean().round(1),
    "mean_field_size_per_race":     decade_field["mean_field"],
    "mean_constructors_per_season": _by_decade(constructors_per_year).mean().round(1),
    "mean_top1_win_share":          _by_decade(top1_share).mean().round(3),
    "mean_top3_win_share":          _by_decade(top3_share).mean().round(3),
})
decade_summary.index.name = "decade"
decade_summary.to_csv(OUT_DIR / "scope_by_decade.csv")

print(f"Plot : {fig_path}")
print(f"Table: {OUT_DIR / 'scope_by_decade.csv'}")
