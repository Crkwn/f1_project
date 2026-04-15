"""
EDA 05 — Teammate comparisons and driver-team switches.

Answers:
  Q11. Within a team-year (same car, same garage, same strategy budget), how
       much do the two teammates differ in performance? This is the cleanest
       separation of 'driver' from 'car' we'll get.
  Q12. When a driver switches teams between seasons, does their performance
       margin relative to their teammate carry over? If yes, 'ability' is a
       persistent driver property; if no, most of it was the previous car.

Metric used:
  For races where BOTH teammates are classified finishers, we compute:
    - finish_gap = driver_A_finish − driver_B_finish (mean across shared races)
    - winshare   = fraction of shared races where driver_A finished ahead

  We canonicalize driver_A as the lower driverId (alphabetical stability),
  so positive gap / winshare < 0.5 means driver B outperformed.

Caveats we should keep in mind:
  - 'Same car' isn't strictly true: team-mate #1 often gets preferential
    strategy, latest upgrades, better side of the garage on set-up. Modern
    top teams especially have asymmetric upgrade paths.
  - DNFs are filtered out here — so the gap metric ignores races where one
    car retired. That's by design for a clean driver-vs-driver comparison,
    but it means we shouldn't interpret the gap as overall ability either.

Outputs:
  stdout : distribution by era, some named extremes, switch residuals
  png    : reports/eda/05_teammate_and_switches/teammate_and_switches.png
  csv    : teammate_pairs.csv, driver_switches.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import DATA_RAW

OUT_DIR = Path(__file__).resolve().parents[2] / "reports" / "eda" / "05_teammate_and_switches"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
races        = pd.read_csv(DATA_RAW / "races.csv")
results      = pd.read_csv(DATA_RAW / "results.csv")
drivers      = pd.read_csv(DATA_RAW / "drivers.csv")
constructors = pd.read_csv(DATA_RAW / "constructors.csv")

res = results.merge(races[["raceId", "year"]], on="raceId")
res = res.merge(drivers[["driverId", "forename", "surname"]], on="driverId")
res["driver_name"] = res["forename"] + " " + res["surname"]
res = res.merge(
    constructors[["constructorId", "name"]].rename(columns={"name": "constructor_name"}),
    on="constructorId",
)

def _is_classified(pt) -> bool:
    try: int(pt); return True
    except: return False

res["classified"] = res["positionText"].apply(_is_classified)
res["finish_pos"] = pd.to_numeric(res["positionText"], errors="coerce")

def era_label(y: int) -> str:
    if y < 1980: return "pre-1980"
    if y < 2000: return "1980–1999"
    if y < 2014: return "2000–2013"
    return "2014–present"

# ---------------------------------------------------------------------------
# Q11. Teammate head-to-head
# ---------------------------------------------------------------------------
# Build (year, constructor, driver) starts table
td_starts = (
    res.groupby(["year", "constructorId", "driverId", "driver_name"])
       .size().reset_index(name="starts")
)
# Keep team-years with exactly 2 drivers, each with ≥5 starts
sizes = td_starts.groupby(["year", "constructorId"]).size()
valid_team_years = sizes[sizes == 2].index

pairs = []
for (yr, cid) in valid_team_years:
    rows = td_starts[(td_starts["year"] == yr) & (td_starts["constructorId"] == cid)]
    if (rows["starts"] >= 5).all():
        # Canonical driver order: lower driverId first
        rows = rows.sort_values("driverId")
        d1_id, d1_name = rows.iloc[0]["driverId"], rows.iloc[0]["driver_name"]
        d2_id, d2_name = rows.iloc[1]["driverId"], rows.iloc[1]["driver_name"]
        # Fetch shared-race finishes
        sub = res[(res["year"] == yr) & (res["constructorId"] == cid)
                  & (res["driverId"].isin([d1_id, d2_id])) & (res["classified"])]
        by_race = sub.pivot_table(index="raceId", columns="driverId",
                                   values="finish_pos", aggfunc="first")
        # Guard: if one driver has no classified finishes at all this year,
        # their column never appears — skip this team-year entirely.
        if d1_id not in by_race.columns or d2_id not in by_race.columns:
            continue
        by_race = by_race[[d1_id, d2_id]].dropna()
        if len(by_race) >= 3:
            gap = (by_race[d1_id] - by_race[d2_id]).mean()
            winshare_d1 = (by_race[d1_id] < by_race[d2_id]).mean()
            pairs.append({
                "year": yr, "constructor_id": cid,
                "constructor": constructors.loc[constructors["constructorId"] == cid, "name"].iloc[0],
                "d1_id": d1_id, "d1_name": d1_name,
                "d2_id": d2_id, "d2_name": d2_name,
                "shared_races": len(by_race),
                "gap_mean": gap,              # >0 means d1 finishes worse than d2
                "winshare_d1": winshare_d1,   # fraction of shared races d1 beat d2
                "era": era_label(yr),
            })

pairs = pd.DataFrame(pairs)

# "Lopsidedness" metrics (symmetric)
pairs["abs_gap"]       = pairs["gap_mean"].abs()
pairs["lopsidedness"]  = (pairs["winshare_d1"] - 0.5).abs()

print("=" * 60)
print("Q11. Teammate head-to-head — both-classified races only")
print("=" * 60)
print(f"Clean team-years (2 drivers, ≥5 starts each, ≥3 shared finishes): {len(pairs)}")
print()
print("Mean absolute finish-position gap between teammates — by era:")
era_stats = pairs.groupby("era").agg(
    n=("abs_gap", "size"),
    mean_abs_gap=("abs_gap", "mean"),
    median_abs_gap=("abs_gap", "median"),
    mean_lopsidedness=("lopsidedness", "mean"),
).round(3)
print(era_stats.to_string())
print()
print("Interpretation:")
print("  abs_gap = average positions separating the two when both finished.")
print("  lopsidedness = |winshare − 0.5|; 0.0 is a 50/50 split, 0.5 is one-sided.")
print()

print("Most lopsided teammate pairings (all eras, top 10):")
top_lop = pairs.sort_values("lopsidedness", ascending=False).head(10)
print(top_lop[["year", "constructor", "d1_name", "d2_name",
               "shared_races", "gap_mean", "winshare_d1"]].to_string(index=False))
print()
print("Most balanced teammate pairings (≥10 shared races, top 10):")
bal = pairs[pairs["shared_races"] >= 10].sort_values("lopsidedness").head(10)
print(bal[["year", "constructor", "d1_name", "d2_name",
           "shared_races", "gap_mean", "winshare_d1"]].to_string(index=False))
print()

# ---------------------------------------------------------------------------
# Q12. Driver-team switches
# ---------------------------------------------------------------------------
# For each driver, find consecutive years with different constructors.
# Ignore mid-season changes (just use primary constructor = most starts).
primary_team = (
    res.groupby(["year", "driverId", "constructorId"]).size()
       .reset_index(name="starts")
       .sort_values(["year", "driverId", "starts"], ascending=[True, True, False])
       .groupby(["year", "driverId"]).head(1)
       .drop(columns=["starts"])
)

# Driver signed margin vs teammate per year (in seasons where pair metric exists)
# From pairs: for each (year, constructor), we know d1 and d2 and gap_mean.
# Driver's "margin vs teammate" = -(gap) if they're d1 (since d1-d2 negative means d1 ahead),
# or +gap if they're d2.
#
# Rather than rebuild here, let's recompute per-driver "margin vs teammate" directly.
driver_year_margin = []
for _, row in pairs.iterrows():
    yr, cid = row["year"], row["constructor_id"]
    d1, d2 = row["d1_id"], row["d2_id"]
    gap = row["gap_mean"]
    # margin = signed "my mean finish − teammate mean finish", so negative = better
    driver_year_margin.append({"year": yr, "constructorId": cid, "driverId": d1,
                                "teammate_id": d2, "margin": gap,
                                "shared_races": row["shared_races"]})
    driver_year_margin.append({"year": yr, "constructorId": cid, "driverId": d2,
                                "teammate_id": d1, "margin": -gap,
                                "shared_races": row["shared_races"]})
dym = pd.DataFrame(driver_year_margin)

# Now build consecutive-year switches
dym_sorted = dym.sort_values(["driverId", "year"])
switches = []
for did, g in dym_sorted.groupby("driverId"):
    g = g.reset_index(drop=True)
    for i in range(len(g) - 1):
        y_prev, y_next = g.iloc[i], g.iloc[i + 1]
        if y_next["year"] - y_prev["year"] == 1 and y_prev["constructorId"] != y_next["constructorId"]:
            switches.append({
                "driverId":   did,
                "year_before":  int(y_prev["year"]),
                "year_after":   int(y_next["year"]),
                "team_before":  int(y_prev["constructorId"]),
                "team_after":   int(y_next["constructorId"]),
                "margin_before": y_prev["margin"],
                "margin_after":  y_next["margin"],
                "shared_before": int(y_prev["shared_races"]),
                "shared_after":  int(y_next["shared_races"]),
            })
switches = pd.DataFrame(switches)
# Attach names for readability
name_map = drivers.set_index("driverId").apply(lambda r: f"{r['forename']} {r['surname']}", axis=1)
team_map = constructors.set_index("constructorId")["name"]
switches["driver"]      = switches["driverId"].map(name_map)
switches["team_before_name"] = switches["team_before"].map(team_map)
switches["team_after_name"]  = switches["team_after"].map(team_map)

print("=" * 60)
print("Q12. Driver team switches — does margin-vs-teammate carry over?")
print("=" * 60)
print(f"Switches with clean teammate-pair metric on both sides: {len(switches)}")
print()

# Correlation of margin_before vs margin_after
if len(switches) >= 10:
    r_pearson = switches[["margin_before", "margin_after"]].corr().iloc[0, 1]
    r_spearman = switches[["margin_before", "margin_after"]].corr(method="spearman").iloc[0, 1]
    print(f"Correlation of teammate-margin before vs after switch:")
    print(f"  Pearson  r = {r_pearson:.3f}   (linear)")
    print(f"  Spearman r = {r_spearman:.3f}   (rank)")
    print("  r > 0 → drivers who beat their teammate at team A also beat their "
          "teammate at team B (ability is persistent).")
    print()

# Notable examples across eras
print("Switches with largest |margin| before — are they still 'stars' after? (top 10)")
print(switches.sort_values("margin_before", key=lambda s: s.abs(), ascending=False)
       .head(10)[["driver", "year_before", "team_before_name", "margin_before",
                  "year_after", "team_after_name", "margin_after"]].to_string(index=False))
print()

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Panel 1: abs gap distribution by era
eras_in_order = ["pre-1980", "1980–1999", "2000–2013", "2014–present"]
data = [pairs[pairs["era"] == e]["abs_gap"].values for e in eras_in_order]
axes[0].boxplot(data, labels=eras_in_order, showfliers=False)
axes[0].set_ylabel("|mean finish gap between teammates|  (positions)")
axes[0].set_title("Teammate gap distribution, by era\n(races where both classified)")
axes[0].grid(alpha=0.3)

# Panel 2: switch margin scatter
axes[1].scatter(switches["margin_before"], switches["margin_after"],
                alpha=0.55, s=22, c="tab:purple")
axes[1].axhline(0, color="gray", linewidth=0.5)
axes[1].axvline(0, color="gray", linewidth=0.5)
lim = max(switches["margin_before"].abs().max(), switches["margin_after"].abs().max())
axes[1].plot([-lim, lim], [-lim, lim], color="gray", linestyle="--", linewidth=0.7,
             label="margin_before = margin_after")
axes[1].set_xlabel("Teammate margin, season BEFORE switch")
axes[1].set_ylabel("Teammate margin, season AFTER switch")
axes[1].set_title("Driver ability persistence across team switches\n(negative margin = driver faster than teammate)")
axes[1].legend(loc="upper left", fontsize=8)
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig_path = OUT_DIR / "teammate_and_switches.png"
fig.savefig(fig_path, dpi=120)
plt.close(fig)

pairs.to_csv(OUT_DIR / "teammate_pairs.csv", index=False)
switches.to_csv(OUT_DIR / "driver_switches.csv", index=False)

print(f"Plot : {fig_path}")
print(f"Table: {OUT_DIR / 'teammate_pairs.csv'}")
print(f"Table: {OUT_DIR / 'driver_switches.csv'}")
