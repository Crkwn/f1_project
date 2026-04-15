"""
EDA 02 — Outcome distributions.

Answers:
  Q4. Distribution of finishing position per race — how many of a race's starters
      end up classified vs retired, and how the attrition per race is distributed.
  Q5. Distribution of wins across drivers — how many distinct drivers win a race
      in a given season, and how many drivers ever win in their career.
  Q6. Win concentration at the *driver* level (as opposed to constructor-level
      in Q3) — what share of a season's wins are held by the top-1 / top-3
      drivers, and how that has moved over time.

Outputs:
  stdout : summary tables by decade, plus headline numbers and recent anchors
  png    : reports/eda/02_outcomes/outcomes_by_season.png
  csv    : reports/eda/02_outcomes/outcomes_by_decade.csv
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import DATA_RAW

OUT_DIR = Path(__file__).resolve().parents[2] / "reports" / "eda" / "02_outcomes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
races   = pd.read_csv(DATA_RAW / "races.csv")
results = pd.read_csv(DATA_RAW / "results.csv")
drivers = pd.read_csv(DATA_RAW / "drivers.csv")

res = results.merge(races[["raceId", "year"]], on="raceId")
res = res.merge(
    drivers[["driverId", "forename", "surname"]], on="driverId"
)
res["driver_name"] = res["forename"] + " " + res["surname"]


def _by_decade(s: pd.Series) -> pd.DataFrame:
    """Group a year-indexed series into 10-year buckets."""
    return s.groupby((s.index // 10) * 10)


def _is_classified(pt) -> bool:
    """Kaggle uses positionText: numbers = classified, 'R'/'D'/'W'/'N'/'F' = not."""
    try:
        int(pt)
        return True
    except (ValueError, TypeError):
        return False


res["classified"] = res["positionText"].apply(_is_classified)

# ---------------------------------------------------------------------------
# Q4. Finishing position distribution
# ---------------------------------------------------------------------------
per_race = (
    res.groupby("raceId")
       .agg(starters=("driverId", "count"),
            classified=("classified", "sum"))
       .reset_index()
)
per_race["dnfs"] = per_race["starters"] - per_race["classified"]
per_race["classified_frac"] = per_race["classified"] / per_race["starters"]
per_race = per_race.merge(races[["raceId", "year"]], on="raceId")

classified_frac_by_year = per_race.groupby("year")["classified_frac"].mean()
dnfs_per_race_by_year   = per_race.groupby("year")["dnfs"].mean()

print("=" * 60)
print("Q4. Finishing-position distribution (classified vs retired)")
print("=" * 60)
print("Averaged within each race then across the decade:")
q4_decade = pd.DataFrame({
    "mean_starters":        _by_decade(per_race.groupby("year")["starters"].mean()).mean().round(1),
    "mean_classified":      _by_decade(per_race.groupby("year")["classified"].mean()).mean().round(1),
    "mean_dnfs_per_race":   _by_decade(dnfs_per_race_by_year).mean().round(2),
    "classified_frac":      _by_decade(classified_frac_by_year).mean().round(3),
})
print(q4_decade.to_string())
print()

# ---------------------------------------------------------------------------
# Q5. Distribution of wins across drivers
# ---------------------------------------------------------------------------
# positionOrder=1 is always the winner (kaggle fills DNFs after real positions)
wins = res[res["positionOrder"] == 1]
wins_per_year_driver = wins.groupby(["year", "driverId"]).size().reset_index(name="wins")

unique_winners_per_year = wins_per_year_driver.groupby("year")["driverId"].nunique()
races_per_year          = races.groupby("year").size()
win_diversity           = (unique_winners_per_year / races_per_year).rename("winners_per_race")

print("=" * 60)
print("Q5. How many distinct drivers win each season")
print("=" * 60)
q5_decade = pd.DataFrame({
    "mean_unique_winners": _by_decade(unique_winners_per_year).mean().round(1),
    "mean_races":          _by_decade(races_per_year).mean().round(1),
    "winners_per_race":    _by_decade(win_diversity).mean().round(3),
})
print(q5_decade.to_string())
print()

career_wins = wins.groupby("driverId").size().sort_values(ascending=False)
all_drivers_count = res["driverId"].nunique()
print(f"Drivers who won ≥1 race, 1950–2024: {len(career_wins)} / {all_drivers_count} "
      f"({len(career_wins)/all_drivers_count:.1%})")
print("\nCareer-win distribution (counts of drivers at each win total):")
bucket = career_wins.value_counts().sort_index()
for w in [1, 2, 3, 5, 10, 20, 50, 100]:
    n = (career_wins >= w).sum()
    print(f"  ≥ {w:>3} wins : {n:>3} drivers")
print()

top10 = (
    drivers[["driverId", "forename", "surname"]]
    .merge(career_wins.rename("wins"), left_on="driverId", right_index=True)
    .sort_values("wins", ascending=False).head(10)
)
top10["driver"] = top10["forename"] + " " + top10["surname"]
print("Top 10 drivers by career wins:")
print(top10[["driver", "wins"]].to_string(index=False))
print()

# ---------------------------------------------------------------------------
# Q6. Win concentration at the driver level
# ---------------------------------------------------------------------------
top1_driver = (
    wins_per_year_driver.sort_values(["year", "wins"], ascending=[True, False])
                        .groupby("year").head(1)
                        .groupby("year")["wins"].sum()
)
top3_driver = (
    wins_per_year_driver.sort_values(["year", "wins"], ascending=[True, False])
                        .groupby("year").head(3)
                        .groupby("year")["wins"].sum()
)
top1_driver_share = (top1_driver / races_per_year).rename("top1_driver_share")
top3_driver_share = (top3_driver / races_per_year).rename("top3_driver_share")

print("=" * 60)
print("Q6. Driver-level win concentration (share of season wins)")
print("=" * 60)
q6_decade = pd.DataFrame({
    "top1_driver_share": _by_decade(top1_driver_share).mean().round(3),
    "top3_driver_share": _by_decade(top3_driver_share).mean().round(3),
})
print(q6_decade.to_string())
print()

recent = wins_per_year_driver[wins_per_year_driver["year"] >= 2015]
recent = (
    recent.sort_values(["year", "wins"], ascending=[True, False])
          .groupby("year").head(1)
          .merge(drivers[["driverId", "forename", "surname"]], on="driverId")
          .merge(races_per_year.rename("races").reset_index(), on="year")
)
recent["driver"] = recent["forename"] + " " + recent["surname"]
recent["share"] = (recent["wins"] / recent["races"]).round(3)
print("Top-winning driver per season, 2015 → latest:")
print(recent[["year", "driver", "wins", "races", "share"]].to_string(index=False))
print()

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

axes[0].plot(classified_frac_by_year.index, classified_frac_by_year.values,
             marker=".", linewidth=1)
axes[0].set_ylabel("Fraction classified")
axes[0].set_title("Outcome shape by season, 1950–present")
axes[0].set_ylim(0, 1.05)
axes[0].axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
axes[0].grid(alpha=0.3)

axes[1].plot(unique_winners_per_year.index, unique_winners_per_year.values,
             marker=".", linewidth=1, label="Unique winners / season")
axes[1].plot(races_per_year.index, races_per_year.values,
             marker=".", linewidth=1, label="Races / season", alpha=0.5)
axes[1].set_ylabel("Count")
axes[1].legend(loc="upper left")
axes[1].grid(alpha=0.3)

axes[2].plot(top1_driver_share.index, top1_driver_share.values,
             marker=".", linewidth=1, label="Top-1 driver share of wins")
axes[2].plot(top3_driver_share.index, top3_driver_share.values,
             marker=".", linewidth=1, label="Top-3 driver share of wins")
axes[2].set_ylabel("Fraction of season wins")
axes[2].set_ylim(0, 1.05)
axes[2].set_xlabel("Year")
axes[2].legend(loc="lower right")
axes[2].grid(alpha=0.3)

fig.tight_layout()
fig_path = OUT_DIR / "outcomes_by_season.png"
fig.savefig(fig_path, dpi=120)
plt.close(fig)

# Combined decade CSV
decade_summary = pd.concat([q4_decade, q5_decade, q6_decade], axis=1)
decade_summary.index.name = "decade"
decade_summary.to_csv(OUT_DIR / "outcomes_by_decade.csv")

print(f"Plot : {fig_path}")
print(f"Table: {OUT_DIR / 'outcomes_by_decade.csv'}")
