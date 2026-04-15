"""
EDA 08 — Race-pace residual distribution (the "what shape is the noise" check).

Answers:
  Q17. What shape is the race-day performance noise? The rating-system
       likelihood we plan to plug in carries an implicit assumption about
       this shape:
         Plackett-Luce (PL)         ⇔  Gumbel-distributed performance noise
         Thurstone-Mosteller (TM)   ⇔  Gaussian-distributed performance noise
         heavier-tailed alternatives⇔  Student-t / Fréchet
       Before committing to PL we'd like to know whether real F1 race-pace
       residuals look Gumbel, Gaussian, or heavier-tailed. This is the same
       sanity check a finance model would run before picking a return
       distribution (log-normal vs Student-t vs jump-diffusion).

Approach:
  Scope 2014–present (matches our Stage 1 rating scope).
  For each (driver, race):
    1. Take all lap times. Drop laps > 107% of driver's in-race median
       (the F1 "107% rule" — cleans out pit stops, safety-car laps, etc.).
    2. Require ≥20 clean laps after filtering (≈1/3 race distance).
    3. Driver's race pace = median of clean lap times (ms).
  For each race:
    Driver's pace GAP = (driver median) − (fastest median in that race),
    expressed as % of leader pace. Taking the gap to the RACE LEADER is
    safety-car invariant and circuit-length invariant.
  For each driver with ≥10 races in window:
    Residual = race pace gap − driver's own mean pace gap
    z = residual / driver's own std of pace gaps
  Pool z across all drivers and characterise:
    mean, std, skewness, kurtosis
    fit Gaussian and Gumbel_L, compare log-likelihoods
    QQ plots.

What this test DOES NOT do:
  - Check whether noise is correlated with circuit/weather/era (it may
    be conditionally Gaussian and marginally fat-tailed via mixing).
  - Check driver-level mean skill — that's what the rating will estimate;
    we're asking only about the SHAPE of the residual cloud around each
    driver's own mean.

Outputs:
  stdout : summary stats + log-likelihood comparison + interpretation
  png    : reports/eda/08_pace_distribution/pace_distribution.png
  csv    : driver_race_pace.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import DATA_RAW

OUT_DIR = Path(__file__).resolve().parents[2] / "reports" / "eda" / "08_pace_distribution"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
races      = pd.read_csv(DATA_RAW / "races.csv")
lap_times  = pd.read_csv(DATA_RAW / "lap_times.csv")
drivers    = pd.read_csv(DATA_RAW / "drivers.csv")

modern_race_ids = races[races["year"] >= 2014]["raceId"].unique()
laps = lap_times[lap_times["raceId"].isin(modern_race_ids)].copy()
print(f"Modern-era (2014+) lap rows loaded: {len(laps):,}")

# ---------------------------------------------------------------------------
# Per (driver, race) race pace — clean-lap median, 107% rule
# ---------------------------------------------------------------------------
def race_pace(ms_values: np.ndarray) -> float:
    ms = ms_values[~np.isnan(ms_values)]
    if len(ms) < 20:
        return np.nan
    med = np.median(ms)
    clean = ms[ms <= 1.07 * med]
    if len(clean) < 20:
        return np.nan
    return float(np.median(clean))

# Use transform-free groupby for speed
dr = (laps.groupby(["raceId", "driverId"])["milliseconds"]
          .apply(lambda s: race_pace(s.values))
          .rename("median_pace_ms")
          .reset_index()
          .dropna(subset=["median_pace_ms"]))
print(f"Driver-race rows with ≥20 clean laps: {len(dr):,}")

# Attach year for era splitting later if we want it
dr = dr.merge(races[["raceId", "year"]], on="raceId")
dr = dr.merge(drivers[["driverId", "forename", "surname"]], on="driverId")
dr["driver"] = dr["forename"] + " " + dr["surname"]

# ---------------------------------------------------------------------------
# Pace gap to race leader, as % of leader pace
#   (safety-car invariant, circuit-length invariant)
# ---------------------------------------------------------------------------
race_leader = dr.groupby("raceId")["median_pace_ms"].min().rename("leader_pace_ms")
dr = dr.merge(race_leader, on="raceId")
dr["pace_gap_pct"] = 100.0 * (dr["median_pace_ms"] - dr["leader_pace_ms"]) / dr["leader_pace_ms"]

# ---------------------------------------------------------------------------
# Driver-level mean / std pace gap, keep drivers with ≥10 observations
# ---------------------------------------------------------------------------
driver_stats = (dr.groupby("driverId")["pace_gap_pct"]
                  .agg(mean_gap="mean", std_gap="std", n="size")
                  .reset_index())
keep = driver_stats[driver_stats["n"] >= 10]
print(f"Drivers with ≥10 race-pace observations: {len(keep)}")

dr = dr.merge(keep[["driverId", "mean_gap", "std_gap"]], on="driverId", how="inner")
dr["residual"] = dr["pace_gap_pct"] - dr["mean_gap"]
dr["z"]        = dr["residual"] / dr["std_gap"]

residuals = dr["z"].dropna().values
print(f"Pooled standardised residuals: {len(residuals):,}")

# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------
mean_r    = float(np.mean(residuals))
std_r     = float(np.std(residuals))
skew_r    = float(stats.skew(residuals))
kurt_r    = float(stats.kurtosis(residuals, fisher=False))   # Gaussian = 3

print("\n" + "=" * 66)
print("Q17. Distribution of race-pace residuals (z-scored per driver)")
print("=" * 66)
print(f"Pooled n          : {len(residuals):,}")
print(f"Mean              : {mean_r:+.3f}   (≈0 by construction)")
print(f"Std               : {std_r:.3f}    (≈1 by construction)")
print(f"Skewness          : {skew_r:+.3f}")
print(f"Kurtosis (raw)    : {kurt_r:.3f}")
print()
print("Reference values:")
print(f"  Gaussian        : skew = 0.00,   kurtosis = 3.00")
print(f"  Gumbel_L (left) : skew = -1.14,  kurtosis = 5.40")
print(f"  Gumbel_R (right): skew = +1.14,  kurtosis = 5.40")
print()
print("Why Gumbel_L is the relevant reference for PL:")
print("  PL assumes performance = skill + Gumbel noise. 'High performance'")
print("  means fast pace (LOW pace gap). So in pace-gap space the long tail")
print("  sits on the NEGATIVE side (occasional alien-fast races), giving")
print("  a left-skewed residual distribution — Gumbel_L.")
print()

# ---------------------------------------------------------------------------
# Fit Gaussian and Gumbel_L, compare log-likelihood
# ---------------------------------------------------------------------------
loc_n, scale_n     = stats.norm.fit(residuals)
loc_gl, scale_gl   = stats.gumbel_l.fit(residuals)
loc_gr, scale_gr   = stats.gumbel_r.fit(residuals)
# Student-t with free df — a classic fat-tail alternative
df_t, loc_t, scale_t = stats.t.fit(residuals)

ll_gauss  = float(np.sum(stats.norm.logpdf(residuals, loc_n,  scale_n)))
ll_gumbel_l = float(np.sum(stats.gumbel_l.logpdf(residuals, loc_gl, scale_gl)))
ll_gumbel_r = float(np.sum(stats.gumbel_r.logpdf(residuals, loc_gr, scale_gr)))
ll_t      = float(np.sum(stats.t.logpdf(residuals, df_t, loc_t, scale_t)))

print("Maximum-likelihood fits:")
print(f"  Gaussian           μ={loc_n:+.3f}, σ={scale_n:.3f}       loglik = {ll_gauss:>11,.1f}")
print(f"  Gumbel_L (left)    loc={loc_gl:+.3f}, scale={scale_gl:.3f}  loglik = {ll_gumbel_l:>11,.1f}")
print(f"  Gumbel_R (right)   loc={loc_gr:+.3f}, scale={scale_gr:.3f}  loglik = {ll_gumbel_r:>11,.1f}")
print(f"  Student-t          df={df_t:.2f}, loc={loc_t:+.3f}, sc={scale_t:.3f}   loglik = {ll_t:>11,.1f}")
print()
print("Higher loglik = better fit. Δ loglik of ~+2 per ~100 obs is meaningful.")
print(f"  Gumbel_L − Gaussian  : {ll_gumbel_l - ll_gauss:+,.1f}")
print(f"  Student-t − Gaussian : {ll_t      - ll_gauss:+,.1f}")
print(f"  Student-t − Gumbel_L : {ll_t      - ll_gumbel_l:+,.1f}")
print()

# ---------------------------------------------------------------------------
# Named examples — the biggest +ve residuals (worst vs own average)
# and -ve residuals (best vs own average) in the window
# ---------------------------------------------------------------------------
# year is already in dr from an earlier merge; just attach race name
dr_named = dr.merge(races[["raceId", "name"]], on="raceId")
top_bad  = dr_named.sort_values("z", ascending=False).head(8)
top_good = dr_named.sort_values("z", ascending=True).head(8)
print("Most extreme bad-weekend residuals (slower than own mean, top 8):")
print(top_bad[["year", "name", "driver", "pace_gap_pct", "mean_gap", "z"]]
      .round(3).to_string(index=False))
print()
print("Most extreme good-weekend residuals (faster than own mean, top 8):")
print(top_good[["year", "name", "driver", "pace_gap_pct", "mean_gap", "z"]]
      .round(3).to_string(index=False))
print()

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

xs = np.linspace(-5, 5, 400)
axes[0].hist(residuals, bins=80, density=True, alpha=0.55, color="tab:blue",
             label=f"empirical (n={len(residuals):,})")
axes[0].plot(xs, stats.norm.pdf(xs, loc_n, scale_n),
             color="tab:orange", lw=2, label="Gaussian fit")
axes[0].plot(xs, stats.gumbel_l.pdf(xs, loc_gl, scale_gl),
             color="tab:red", lw=2, linestyle="--", label="Gumbel_L fit")
axes[0].plot(xs, stats.t.pdf(xs, df_t, loc_t, scale_t),
             color="tab:green", lw=2, linestyle=":", label=f"Student-t fit (df={df_t:.1f})")
axes[0].set_xlim(-5, 5)
axes[0].set_xlabel("z-scored pace residual (driver's own mean = 0)")
axes[0].set_ylabel("density")
axes[0].set_title(f"Race-pace residuals\nskew={skew_r:+.2f}  kurtosis={kurt_r:.2f}")
axes[0].legend(fontsize=8, loc="upper right")
axes[0].grid(alpha=0.3)

# QQ plot vs Gaussian
stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].get_lines()[0].set_markersize(2.5)
axes[1].get_lines()[0].set_alpha(0.4)
axes[1].set_title("QQ plot vs Gaussian")
axes[1].grid(alpha=0.3)

# QQ plot vs Gumbel_L
stats.probplot(residuals, dist="gumbel_l", plot=axes[2])
axes[2].get_lines()[0].set_markersize(2.5)
axes[2].get_lines()[0].set_alpha(0.4)
axes[2].set_title("QQ plot vs Gumbel_L")
axes[2].grid(alpha=0.3)

fig.tight_layout()
fig_path = OUT_DIR / "pace_distribution.png"
fig.savefig(fig_path, dpi=120)
plt.close(fig)

dr_out = dr[["raceId", "year", "driverId", "driver", "median_pace_ms",
             "leader_pace_ms", "pace_gap_pct", "mean_gap", "std_gap",
             "residual", "z"]]
dr_out.to_csv(OUT_DIR / "driver_race_pace.csv", index=False)

print(f"Plot : {fig_path}")
print(f"Table: {OUT_DIR / 'driver_race_pace.csv'}")
