"""
EDA 09 — Team resources proxy: does previous-year constructor championship
          rank predict pole-position rate this year?

Context:
  Team budget would be a real structural factor in F1 but is not in our
  data. Ergast / Kaggle has constructor IDs, names, and championship
  points — no spend figures. Budget figures are only publicly precise
  for the budget-cap era (2021+); earlier years exist only as journalist
  estimates and would need manual compilation.

  For data-only analysis we use a proxy: the team's final championship
  rank in the PREVIOUS season. This is the strongest ex-ante measure of
  team strength available inside Ergast. The correlation it gives is an
  UPPER BOUND on the marginal budget effect, because last-year's rank
  already absorbs budget, engineering quality, driver quality, org
  continuity, and manufacturing depth. Teasing out a pure budget effect
  would require external budget data and residualising rank on spend.

Scope:
  Qualifying records are consistent from 1994+. We run the analysis
  on 2000+ for stability with our other EDA scopes.

Caveats to keep in mind:
  - Rank proxy is circular: good last year → likely good this year.
  - Rebrands / ownership changes blur the signal in transition years
    (Aston Martin/Racing Point, Alpine/Renault, RB/AlphaTauri). We
    include these at face value — no manual fixing.
  - New entries (no previous-year rank) are excluded from the rank
    analysis; they're aggregated separately.

Outputs:
  stdout : per-rank pole rates, Spearman correlation, era splits
  png    : reports/eda/09_team_resources/team_resources.png
  csv    : constructor_year_pole_rates.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import DATA_RAW

OUT_DIR = Path(__file__).resolve().parents[2] / "reports" / "eda" / "09_team_resources"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
races             = pd.read_csv(DATA_RAW / "races.csv")
qualifying        = pd.read_csv(DATA_RAW / "qualifying.csv")
results           = pd.read_csv(DATA_RAW / "results.csv")
constructors      = pd.read_csv(DATA_RAW / "constructors.csv")
constructor_stand = pd.read_csv(DATA_RAW / "constructor_standings.csv")

# ---------------------------------------------------------------------------
# Previous-year constructor rank
#   final-race-of-season standing for each (year, constructor)
# ---------------------------------------------------------------------------
# Last race of each year
last_race = races.sort_values(["year", "round"]).groupby("year").tail(1)[["year", "raceId"]]
final_stand = constructor_stand.merge(last_race, on="raceId")
final_stand = final_stand[["year", "constructorId", "position", "points"]].rename(
    columns={"position": "rank_end", "points": "points_end"})

# prev_year rank — shift by 1 year
prev = final_stand.copy()
prev["year"] = prev["year"] + 1
prev = prev.rename(columns={"rank_end": "rank_prev", "points_end": "points_prev"})
prev = prev[["year", "constructorId", "rank_prev", "points_prev"]]

# ---------------------------------------------------------------------------
# Pole counts and races entered per (year, constructor)
# ---------------------------------------------------------------------------
qual = qualifying.merge(races[["raceId", "year"]], on="raceId")
qual = qual[qual["year"] >= 2000]

poles = (qual[qual["position"] == 1]
         .groupby(["year", "constructorId"]).size()
         .reset_index(name="poles"))

# Races entered: count unique raceIds per (year, constructor) in RESULTS
res_yr = results.merge(races[["raceId", "year"]], on="raceId")
res_yr = res_yr[res_yr["year"] >= 2000]
entries = (res_yr.groupby(["year", "constructorId"])["raceId"]
           .nunique().reset_index(name="races"))

cy = entries.merge(poles, on=["year", "constructorId"], how="left")
cy["poles"] = cy["poles"].fillna(0).astype(int)
cy["pole_rate"] = cy["poles"] / cy["races"]

# Attach rank_prev
cy = cy.merge(prev, on=["year", "constructorId"], how="left")
cy = cy.merge(constructors[["constructorId", "name"]], on="constructorId")

# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------
print("=" * 66)
print("Q18. Previous-year constructor rank → this year's pole rate")
print("=" * 66)
print(f"Total (year, constructor) cells, 2000+: {len(cy)}")
print(f"With a previous-year rank (i.e. team existed last year): "
      f"{cy['rank_prev'].notna().sum()}")
print()

have_prev = cy.dropna(subset=["rank_prev"]).copy()
have_prev["rank_prev"] = have_prev["rank_prev"].astype(int)

# Bin by rank_prev — top 3, 4–7, 8–10, 11+
def rank_bin(r):
    if r <= 3: return "1–3"
    if r <= 7: return "4–7"
    if r <= 10: return "8–10"
    return "11+"

have_prev["rank_bin"] = have_prev["rank_prev"].apply(rank_bin)

bin_order = ["1–3", "4–7", "8–10", "11+"]
summary = (have_prev.groupby("rank_bin")
                    .agg(n_cells=("pole_rate", "size"),
                         total_races=("races", "sum"),
                         total_poles=("poles", "sum"),
                         mean_pole_rate=("pole_rate", "mean"),
                         median_pole_rate=("pole_rate", "median"))
                    .reindex(bin_order).round(3))
summary["empirical_pole_rate"] = (summary["total_poles"] / summary["total_races"]).round(3)
print("Pole rate by previous-year championship rank (2000+):")
print(summary.to_string())
print()

# Correlation
rho, p = stats.spearmanr(have_prev["rank_prev"], have_prev["pole_rate"])
print(f"Spearman correlation (rank_prev vs pole_rate) : {rho:+.3f}   (p={p:.2g})")
print("  Negative ρ expected: lower (better) rank → higher pole rate.")
print()

# Era split
def era(y):
    if y < 2014: return "2000–2013"
    if y < 2022: return "2014–2021"
    return "2022–present"

have_prev["era"] = have_prev["year"].apply(era)
print("Spearman(rank_prev, pole_rate) by era:")
for e in ["2000–2013", "2014–2021", "2022–present"]:
    sub = have_prev[have_prev["era"] == e]
    if len(sub) > 5:
        r, _ = stats.spearmanr(sub["rank_prev"], sub["pole_rate"])
        print(f"  {e} (n={len(sub):>3}) : ρ = {r:+.3f}")
print()

# Concentration: how many teams captured, say, 90% of poles 2014+?
mod = have_prev[have_prev["year"] >= 2014].sort_values("total_poles" if "total_poles" in have_prev else "poles", ascending=False)
by_team = (have_prev[have_prev["year"] >= 2014]
           .groupby("name")
           .agg(poles=("poles", "sum"), races=("races", "sum"))
           .sort_values("poles", ascending=False))
by_team["share_of_poles"] = (by_team["poles"] / by_team["poles"].sum()).round(3)
by_team["cumulative_share"] = by_team["share_of_poles"].cumsum().round(3)
print("Pole concentration, 2014–present:")
print(by_team.head(10).to_string())
print()

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Panel 1: scatter rank_prev vs pole_rate (all eras in one)
axes[0].scatter(have_prev["rank_prev"], have_prev["pole_rate"],
                s=25, alpha=0.5, color="tab:blue")
axes[0].set_xlabel("Previous-year constructor rank (1 = champion)")
axes[0].set_ylabel("This-year pole rate")
axes[0].set_title(f"Previous-year rank vs this-year pole rate\nSpearman ρ = {rho:+.3f}")
axes[0].set_xticks(range(1, int(have_prev["rank_prev"].max()) + 1))
axes[0].grid(alpha=0.3)

# Panel 2: pole rate time series for top-name constructors
top_names = by_team.head(6).index.tolist()
for name in top_names:
    sub = have_prev[have_prev["name"] == name].sort_values("year")
    if len(sub) >= 3:
        axes[1].plot(sub["year"], sub["pole_rate"], marker="o",
                     markersize=4, linewidth=1.5, label=name, alpha=0.85)
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Pole rate")
axes[1].set_title("Pole rate over time — top-6 pole teams (2014–present window)")
axes[1].legend(fontsize=8, loc="upper right")
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig_path = OUT_DIR / "team_resources.png"
fig.savefig(fig_path, dpi=120)
plt.close(fig)

cy.to_csv(OUT_DIR / "constructor_year_pole_rates.csv", index=False)

print(f"Plot : {fig_path}")
print(f"Table: {OUT_DIR / 'constructor_year_pole_rates.csv'}")
