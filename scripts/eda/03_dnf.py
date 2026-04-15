"""
EDA 03 — DNFs (Did Not Finish): rates, causes, and where the signal comes from.

Answers:
  Q7. What fraction of starts end in DNF, split by *cause* (mechanical / accident
      / driver-health / admin), and how has this moved 1950 → present?
  Q8. Is DNF rate a property of the car (constructor) or the driver?  We test this
      by looking at how correlated the two teammates' DNF rates are within a
      single team-year.  If the car is the main driver, teammates should have
      similar DNF rates; if it's the driver, they shouldn't.

Status family mapping
  Each of the 140 status codes in status.csv is hand-classified into one of:
    finished     — completed the race (incl. "+N Laps" lapped finishers)
    mechanical   — car failure attributable mostly to the constructor
    accident     — crash, collision, spin, debris — mostly on-track incident
    driver       — driver health / injury (physical, ill, injured, ...)
    disqualified — post-race administrative removal
    other        — withdrew, not classified, did not (pre)qualify, safety, etc.

Outputs:
  stdout : decadal fractions by status family; teammate DNF correlation by era
  png    : reports/eda/03_dnf/dnf_over_time.png
  csv    : reports/eda/03_dnf/status_family_by_decade.csv
  csv    : reports/eda/03_dnf/teammate_dnf_pairs.csv
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import DATA_RAW

OUT_DIR = Path(__file__).resolve().parents[2] / "reports" / "eda" / "03_dnf"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
races        = pd.read_csv(DATA_RAW / "races.csv")
results      = pd.read_csv(DATA_RAW / "results.csv")
status       = pd.read_csv(DATA_RAW / "status.csv")
constructors = pd.read_csv(DATA_RAW / "constructors.csv")

res = results.merge(races[["raceId", "year"]], on="raceId")
res = res.merge(status, on="statusId")
res = res.merge(
    constructors[["constructorId", "constructorRef", "name"]]
        .rename(columns={"name": "constructor_name"}),
    on="constructorId",
)

# ---------------------------------------------------------------------------
# Status family mapping
# ---------------------------------------------------------------------------
MECHANICAL = {
    "Engine", "Gearbox", "Transmission", "Clutch", "Hydraulics", "Electrical",
    "Radiator", "Suspension", "Brakes", "Differential", "Overheating",
    "Mechanical", "Tyre", "Driver Seat", "Puncture", "Driveshaft",
    "Fuel pressure", "Front wing", "Water pressure", "Refuelling", "Wheel",
    "Throttle", "Steering", "Technical", "Electronics", "Broken wing",
    "Heat shield fire", "Exhaust", "Oil leak", "Wheel rim", "Water leak",
    "Fuel pump", "Track rod", "Oil pressure", "Engine fire", "Engine misfire",
    "Tyre puncture", "Out of fuel", "Wheel nut", "Pneumatics", "Handling",
    "Rear wing", "Fire", "Wheel bearing", "Fuel system", "Oil line", "Fuel rig",
    "Launch control", "Fuel", "Power loss", "Vibrations", "Drivetrain",
    "Ignition", "Chassis", "Battery", "Stalled", "Halfshaft", "Crankshaft",
    "Alternator", "Safety belt", "Oil pump", "Fuel leak", "Injection",
    "Distributor", "Turbo", "CV joint", "Water pump", "Spark plugs",
    "Fuel pipe", "Oil pipe", "Axle", "Water pipe", "Magneto", "Supercharger",
    "Power Unit", "ERS", "Brake duct", "Seat", "Undertray", "Cooling system",
}
ACCIDENT = {
    "Accident", "Collision", "Spun off", "Fatal accident", "Collision damage",
    "Damage", "Debris",
}
DRIVER = {
    "Physical", "Injured", "Injury", "Eye injury", "Driver unwell", "Illness",
}
DISQUALIFIED = {"Disqualified", "Excluded", "Underweight"}
OTHER_DNF = {
    "Retired", "Withdrew", "Not classified", "Did not qualify",
    "Did not prequalify", "Not restarted", "Safety", "Safety concerns",
    "107% Rule",
}


def family_of(s: str) -> str:
    if s == "Finished" or s.startswith("+"):
        return "finished"
    if s in MECHANICAL:   return "mechanical"
    if s in ACCIDENT:     return "accident"
    if s in DRIVER:       return "driver"
    if s in DISQUALIFIED: return "disqualified"
    if s in OTHER_DNF:    return "other"
    return "other"


res["family"] = res["status"].apply(family_of)
res["dnf"]    = res["family"] != "finished"

# Sanity check: no un-mapped statuses left unhandled
unmapped = (
    res[res["family"] == "other"]
       .loc[~res["status"].isin(OTHER_DNF)]
       .groupby("status").size().sort_values(ascending=False)
)
if len(unmapped):
    print("WARNING — status codes falling through to 'other':")
    print(unmapped.to_string())
    print()

# ---------------------------------------------------------------------------
# Q7. DNF rate by cause, over time
# ---------------------------------------------------------------------------
by_year_family = (
    res.groupby(["year", "family"]).size()
       .unstack(fill_value=0)
)
for c in ["finished", "mechanical", "accident", "driver", "disqualified", "other"]:
    if c not in by_year_family.columns:
        by_year_family[c] = 0
by_year_family = by_year_family[
    ["finished", "mechanical", "accident", "driver", "disqualified", "other"]
]
year_frac = by_year_family.div(by_year_family.sum(axis=1), axis=0)

# Decadal averages
year_frac["decade"] = (year_frac.index // 10) * 10
dec = year_frac.groupby("decade").mean().round(3)
year_frac = year_frac.drop(columns=["decade"])

print("=" * 60)
print("Q7. Outcome fractions by status family — by decade")
print("=" * 60)
print(dec.to_string())
print()

dnf_rate_by_year = res.groupby("year")["dnf"].mean().rename("dnf_rate")
print("Overall DNF rate by decade (any non-finish):")
print(dnf_rate_by_year.groupby((dnf_rate_by_year.index // 10) * 10).mean().round(3).to_string())
print()

# ---------------------------------------------------------------------------
# Q8. Car vs driver: teammate DNF correlation within team-years
# ---------------------------------------------------------------------------
by_td = (
    res.groupby(["year", "constructorId", "driverId"])
       .agg(starts=("resultId", "count"), dnfs=("dnf", "sum"))
       .reset_index()
)
by_td["dnf_rate"] = by_td["dnfs"] / by_td["starts"]

# Keep only clean team-years: exactly 2 drivers, each with ≥ 5 starts
sizes = by_td.groupby(["year", "constructorId"]).size()
valid = sizes[sizes == 2].index

pairs = []
for (year, cid) in valid:
    tdf = by_td[(by_td["year"] == year) & (by_td["constructorId"] == cid)]
    if (tdf["starts"] >= 5).all():
        d1, d2 = tdf.iloc[0], tdf.iloc[1]
        pairs.append({
            "year": year,
            "constructorId": cid,
            "d1_dnf_rate": d1["dnf_rate"],
            "d2_dnf_rate": d2["dnf_rate"],
            "d1_starts": d1["starts"],
            "d2_starts": d2["starts"],
        })
pairs = pd.DataFrame(pairs)


def era_label(y: int) -> str:
    if y < 1980: return "pre-1980"
    if y < 2000: return "1980–1999"
    if y < 2014: return "2000–2013"
    return "2014–present"


pairs["era"] = pairs["year"].apply(era_label)

print("=" * 60)
print("Q8. Teammate DNF correlation (car vs driver)")
print("=" * 60)
print(f"Clean team-years (2 drivers, each ≥5 starts): {len(pairs)}")
print()
print("Correlation of the two teammates' DNF rates within the same team-year:")
print("  r → 1.0 : DNF is a car thing (both drivers break down together)")
print("  r → 0.0 : DNF is a driver thing (or pure noise)")
print()
for e in ["pre-1980", "1980–1999", "2000–2013", "2014–present"]:
    sub = pairs[pairs["era"] == e]
    if len(sub) > 5:
        r = sub[["d1_dnf_rate", "d2_dnf_rate"]].corr().iloc[0, 1]
        print(f"  {e:<14}  n = {len(sub):>4}   r = {r:.3f}")
print()

# Also: within-team DNF rate range (max - min). If cars matter, the range tells
# us how much within-team-year variation there is beyond the common car failure.
pairs["within_team_gap"] = (pairs["d1_dnf_rate"] - pairs["d2_dnf_rate"]).abs()
print("Mean |driver1 − driver2| DNF-rate gap within team-year, by era:")
print(pairs.groupby("era")["within_team_gap"].mean().round(3).to_string())
print()

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(11, 9))

stack_cols = ["finished", "mechanical", "accident", "driver", "disqualified", "other"]
axes[0].stackplot(year_frac.index, year_frac[stack_cols].T,
                  labels=stack_cols, alpha=0.85)
axes[0].set_ylabel("Fraction of starts")
axes[0].set_title("Outcome families by season, 1950–present")
axes[0].set_ylim(0, 1)
axes[0].legend(loc="center right", fontsize=8)
axes[0].grid(alpha=0.3)
axes[0].set_xlabel("Year")

# Teammate DNF scatter — colour by era so eras separate visually
era_colors = {
    "pre-1980":      "tab:gray",
    "1980–1999":     "tab:blue",
    "2000–2013":     "tab:orange",
    "2014–present":  "tab:red",
}
for e, c in era_colors.items():
    sub = pairs[pairs["era"] == e]
    axes[1].scatter(sub["d1_dnf_rate"], sub["d2_dnf_rate"],
                    alpha=0.5, s=16, c=c, label=f"{e}  (n={len(sub)})")
axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=0.7)
axes[1].set_xlabel("Driver 1 DNF rate (team-year)")
axes[1].set_ylabel("Driver 2 DNF rate (team-year)")
axes[1].set_title("Teammate DNF rates within the same team-year")
axes[1].set_xlim(-0.02, 1.02)
axes[1].set_ylim(-0.02, 1.02)
axes[1].legend(loc="lower right", fontsize=8)
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig_path = OUT_DIR / "dnf_over_time.png"
fig.savefig(fig_path, dpi=120)
plt.close(fig)

dec.to_csv(OUT_DIR / "status_family_by_decade.csv")
pairs.to_csv(OUT_DIR / "teammate_dnf_pairs.csv", index=False)

print(f"Plot : {fig_path}")
print(f"Table: {OUT_DIR / 'status_family_by_decade.csv'}")
print(f"Pairs: {OUT_DIR / 'teammate_dnf_pairs.csv'}")
