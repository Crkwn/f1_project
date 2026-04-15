# F1 win-probability model — progress + research plan

Stopping point: end of v1. Everything below describes what exists, what
the numbers say, and what a research programme on top of it should look
like. Pick up from **§8 Resuming next session** on return.

---

## 1. Status in one paragraph

A three-piece model is fully built and runs end-to-end: (1) per-driver
skill (OpenSkill Plackett–Luce with structural σ inflations for team
switches and inactivity), (2) per-(constructor, year) mechanical
reliability (Beta–Binomial posterior), (3) per-race predictor that
combines driver skill + reliability + prev-year constructor pace +
starting grid into a softmax with a fitted temperature, plus a
Gumbel-max Monte-Carlo sampler for P(podium) and P(points). A
walk-forward back-test on 68 races (2022–2024) gives **39.5% mean
probability on the actual winner** and **52% top-1 hit rate**, measured
against four calibrated baselines. A predict CLI runs on any historical
or hypothetical race. A static demo site in `/docs` can be served on
GitHub Pages.

---

## 2. What's built

| Stage | File | CLI | Status |
|---|---|---|---|
| 1a — Driver rating | `src/model/rating.py` | `scripts/model/build_ratings.py` | working |
| 1b — Constructor reliability | `src/model/reliability.py` | `scripts/model/build_reliability.py` | working |
| 2 — Race predictor | `src/model/race_predictor.py` | `scripts/model/predict_race.py` | working |
| Walk-forward back-test | `scripts/model/backtest.py` | same | working |
| Static demo site | `docs/` + `scripts/web/build_site.py` | same | working |
| Config / levers | `src/config.py` | — | current |

Supporting files:
- `src/model/status_families.py` — classifies Ergast status strings into
  {finished, mechanical, accident, driver, disqualified, other}. Drives
  the censoring for the rating update and the DNF count for reliability.
- `scripts/eda/01_scope.py … 09_team_resources.py` — exploratory
  analysis that produced the feature choices. `09_team_resources.py`
  validated prev-year constructor rank → current-year pace (Spearman
  ρ = −0.60 overall, −0.73 in budget-cap era).

Artefacts on disk:
- `data/processed/rater.pkl` — fully-trained rater (processed through
  the last race in the data; not suitable for predictions on races
  within the training range — see `predict_race.py` which replays).
- `data/processed/reliability.pkl` — same for reliability.
- `reports/backtest/predictions_ordinal.csv` — per-(race, driver)
  predictions + actual outcomes across the 68-race holdout.
- `reports/backtest/metrics_ordinal.{csv,txt}` — headline metrics +
  calibration table + named-race dump.
- `reports/backtest/predictions_z_ordinal.csv` — same for the
  alternative skill-feature mode (worse, retained for comparison).
- `reports/backtest/reliability_ordinal.png` — reliability diagram.
- `reports/predictions/<year>-R<round>_ordinal.csv` — predict CLI output.
- `docs/data.json` — full back-test output packed for the demo site.

---

## 3. How to run each piece

```bash
# Stage 1a — driver rating from scratch
python scripts/model/build_ratings.py

# Stage 1b — constructor reliability
python scripts/model/build_reliability.py

# Walk-forward back-test (writes predictions_<mode>.csv, metrics, PNG)
python scripts/model/backtest.py                  # mode=ordinal (default)
python scripts/model/backtest.py --mode z_ordinal # alternative
python scripts/model/backtest.py --fit-tau        # grid-search τ
python scripts/model/backtest.py --holdout-start 2023  # shorter holdout

# Predict a single race (historical — replays state so no look-ahead)
python scripts/model/predict_race.py --latest
python scripts/model/predict_race.py --year 2024 --round 22
python scripts/model/predict_race.py --latest --overround 1.20  # + book odds

# Predict a hypothetical grid you hand it
python scripts/model/predict_race.py --year 2025 --field my_field.csv

# Rebuild the demo site data (after any back-test re-run)
python scripts/web/build_site.py

# Preview the demo locally
python3 -m http.server 8765 --directory docs     # open http://localhost:8765
```

---

## 4. Headline numbers we actually verified

Holdout: 68 races, 2022–2024. Training window: 2014–2021.

**Probability assigned to the actual winner** (higher = sharper model; random = 5%):

| Method | P(actual winner) | top-1 | top-3 | avg winner rank |
|---|---|---|---|---|
| Model (analytic A) | **0.419** | 0.559 | 0.853 | 1.97 |
| Model (MC B) | **0.395** | 0.515 | 0.838 | 2.06 |
| Baseline: grid-softmax | 0.138 | 0.338 | 0.765 | 3.10 |
| Baseline: pole-empirical | 0.282 | 0.515 | 0.941 | 1.96 |
| Baseline: last-race-winner | 0.167 | 0.441 | 0.985 | 1.68 |
| Baseline: championship-leader | 0.243 | 0.603 | 0.971 | 1.63 |

**Hit rate of the model's top pick, conditional on its own confidence:**

| Model's confidence on top pick | Races | Actually won |
|---|---|---|
| 30–40% | 5 | 0% |
| 40–50% | 16 | 44% |
| 50–60% | 6 | 50% |
| 60–70% | 19 | 47% |
| 70–80% | 21 | 71% |
| 80%+ | 1 | 100% |

**Calibration** (all (race, driver) predictions, bucketed):

| Bin | n | Mean predicted | Observed rate |
|---|---|---|---|
| 0.0–0.1 | 1237 | 0.013 | 0.019 |
| 0.1–0.2 | 33 | 0.122 | 0.061 |
| 0.2–0.3 | 11 | 0.268 | 0.273 |
| 0.3–0.4 | 14 | 0.364 | 0.286 |
| 0.4–0.5 | 17 | 0.456 | 0.471 |
| 0.5–0.6 | 6 | 0.552 | 0.500 |
| **0.6–0.7** | **19** | **0.650** | **0.474** ← overconfident |
| 0.7–0.8 | 21 | 0.728 | 0.714 |
| 0.8–0.9 | 1 | 0.806 | 1.000 |

**Mode comparison** (same holdout, two skill-feature modes):

| Mode | P(actual winner) | top-1 |
|---|---|---|
| ordinal (μ − 3σ) | **0.395** | **0.515** |
| z_ordinal (per-race z of ordinal) | 0.262 | 0.426 |

---

## 5. Design decisions locked in (with config refs)

All tunables live in `src/config.py` — each is documented with its
rationale and the alternative we considered.

| Parameter | Value | Decision |
|---|---|---|
| `RATING_START_YEAR` | 2014 | Hybrid-era rule break. Pre-2014 is EDA context only. |
| `RATING_INITIAL_MU` / `SIGMA` | 25 / 8.33 | OpenSkill default priors. Unit-less. |
| `QUALI_UPDATE_WEIGHT` | 1.0 | Quali as a second weekend update, same weight as race. |
| `SIGMA_BUMP_TEAM_SWITCH` | 1.5 | Variance-additive bump at team change. |
| `SIGMA_BUMP_PER_YEAR_INACTIVE` | 0.7 | Variance-additive per full year missed. |
| `SIGMA_FLOOR` | 1.0 | Never shrink σ below this. |
| `EVENT_RESIDUAL_Z_CAP` | 4.0 | Belt-and-braces outlier cap on rating residuals. |
| `ALPHA_PRIOR`, `BETA_PRIOR` | 2.0, 10.0 | Beta prior on mech-DNF rate. Mean 16.7%. |
| Reliability year-reset | on | Each season starts fresh. |
| `DRIVER_SCORE_MODE` | `"ordinal"` | Confirmed empirically beats `z_ordinal` (§4). |
| `SOFTMAX_TAU_INIT` | 20.0 | Grid-fit on back-test; revised from initial guess of 80. |
| `W_DRIVER_ORDINAL` | 1.0 | Ordinal already in "units of μ". |
| `W_PACE_PREV_RANK` | 50.0 | Pace ∈ [0,1]; weight = absolute contribution range. |
| `W_GRID` | 30.0 | Grid ∈ [0,1]; ditto. |
| `MC_SAMPLES` | 5000 | MC std at p=0.1 is ~0.4pp, below any calibration tolerance. |

---

## 6. Research plan — what to explore next, true research style

Each item: **observation** → **hypothesis** → **experiment** → **success
criterion** → **what would change our mind**. Effort is a t-shirt size
(S / M / L). Expected-value column is impact if the hypothesis holds.

### Tier 1 — **ship the v1 properly before doing anything else**

#### 1.1  Post-hoc isotonic calibration layer  (S, high EV)
- **Observation**: 0.6–0.7 bucket is 18pp overconfident (65% predicted,
  47% observed, n=19). 0.1–0.2 also off (12% predicted, 6% observed, n=33).
- **Hypothesis**: overconfidence is systematic (not noise), caused by
  τ-fitting that optimises *average* log-loss but doesn't flatten a
  specific bucket. A monotone recalibration learned from the holdout
  will remove it without retouching any model internals.
- **Experiment**: fit `sklearn.isotonic.IsotonicRegression` on
  (predicted P(win), is_winner) pairs from the holdout. Apply at
  prediction time before reporting probabilities. Re-run the back-test
  with held-out slices (leave-one-race-out) and check calibration.
- **Success**: |mean_pred − observed| < 5pp in every bucket with n ≥ 10.
- **Changes our mind**: calibration still fails on 2025 live data →
  it's not a post-hoc fix, we need to inspect model internals.

#### 1.2  Data refresh pipeline  (M, enabler)
- **Observation**: `data/raw/*.csv` is a static Ergast snapshot. Cannot
  run the CLI on the actual next race.
- **Hypothesis**: Jolpica (Ergast-compatible fork) exposes the same
  schema live. A `scripts/data/refresh_jolpica.py` can append new rows
  without breaking the existing readers.
- **Experiment**: build the refresh script; verify row counts and
  schemas match. Re-run `build_ratings.py` + `build_reliability.py`.
- **Success**: walk-forward back-test output is identical (bit-for-bit)
  on the intersecting date range.
- **Changes our mind**: schema drift between Jolpica and local Ergast
  CSVs requires per-column shims.

#### 1.3  Live prediction routine  (S, enabler)
- **Observation**: calibration is measured on the same 68 races we used
  to fit τ and pick the mode. There is no truly out-of-sample test.
- **Experiment**: every race weekend in 2025, after qualifying, run the
  predict CLI and log the prediction. After the race, append the
  actual. One row per (race, driver). Persist to
  `reports/live/history.csv`.
- **Success**: by mid-2025, ≥ 10 fresh races we never trained on →
  recompute reliability diagram and top-1 rate on live data.
- **Value**: everything below depends on this — without real OOS data
  we are guessing about whether the v1 generalises.

### Tier 2 — **diagnose the misses, pick an upgrade**

#### 2.1  Miss-pattern decomposition  (S, medium EV)
- **Observation**: the five most-wrong races split cleanly: 3 are
  first-time winners from young drivers (Norris Miami, Piastri Hungary,
  Sainz Silverstone), 2 are incumbent-champion DNFs (Verstappen
  Bahrain 22, Australia 24).
- **Hypothesis A**: μ updates too slowly for surging drivers — the
  model's skill feature lags reality on drivers trending up.
- **Hypothesis B**: reliability year-reset is too aggressive — a
  dominant team's early-season reliability dip is invisible to the
  model.
- **Experiment**: across the full holdout, tag each "most-wrong" race
  by type. Count {A, B, neither} distribution. Whichever dominates
  determines which Tier-3 experiment to prioritise.
- **Success**: clear majority (≥ 60%) in one bucket.
- **Changes our mind**: misses are evenly mixed → no single structural
  change will help, must add features instead (go to §2.3).

#### 2.2  Baseline robustness across holdout windows  (S, low EV but high information)
- **Observation**: championship-leader baseline's top-1 rate is 60% in
  2022–2024 (vs our 52%), but was 41% in 2014–2021.
- **Hypothesis**: the leader baseline's advantage in this window is
  champion-dominance, not a real model weakness.
- **Experiment**: run the back-test with {2018–2021, 2016–2019,
  2014–2017} holdouts. Compare model-vs-leader top-1 in each.
- **Success**: leader-vs-model top-1 gap flips sign in ≥ 1 other window.
- **Changes our mind**: leader beats us in every window → model has a
  real structural underconfidence on dominant drivers we need to fix.

### Tier 3 — **targeted modelling upgrades**

Pick based on §2.1's outcome.

#### 3.1  Within-season skill drift  (M, high EV if pattern A wins)
- **Observation (conditional)**: model's top-1 misses cluster on
  first-time winners whose μ lagged their true pace.
- **Hypothesis**: injecting σ based on recent-race residual volatility
  (high residuals → driver is trending, widen σ so weekly updates
  propagate faster) will help the model "catch" a surging driver within
  3–4 races instead of 6–8.
- **Experiment**: add a rolling-residual σ-bump to `_apply_structural_inflation`.
  Back-test. Specifically check the 3 first-time-winner races in §2.1.
- **Success**: top-1 hit rate improves by ≥ 3pp on those 3 races
  without degrading the other 65.
- **Changes our mind**: accuracy on first-time winners improves but
  overall log-loss degrades → too noisy, back out.

#### 3.2  Cross-year reliability smoothing (EWMA)  (M, medium EV if pattern B wins)
- **Observation (conditional)**: misses dominated by incumbent DNFs
  that reliability model didn't flag.
- **Hypothesis**: last year's posterior is informative about this
  year's early-season reliability. Current year-reset discards it.
- **Experiment**: blend last-year posterior into this-year prior:
  `α_this_year = α_prior + λ · α_last_year_posterior`, λ ∈ {0.25, 0.5}.
  Back-test. Check the 2 incumbent-DNF races.
- **Success**: reliability on those races drops by ≥ 5pp pre-race,
  widening the model's predicted field.
- **Changes our mind**: EWMA helps early-season but hurts mid-season →
  time-varying λ (decaying through the season).

#### 3.3  Quali gap as a feature  (M, high EV — always runnable)
- **Observation**: currently the only pace signal from the actual
  weekend is grid position, which is discrete and penalised-grid-noisy.
- **Hypothesis**: quali gap to pole (fractional seconds → [0, 1] score)
  is a finer pre-race pace signal and might subsume grid as a feature.
- **Experiment**: add `quali_gap_score = exp(−k · gap_s)` as a 4th
  feature in the combined_score. Refit τ and weights. Back-test.
- **Success**: overall P(actual winner) improves by ≥ 2pp.
- **Changes our mind**: quali gap completely overlaps grid (r > 0.95) →
  drop it, grid was enough.

### Tier 4 — **speculative / defer until Tier 1–3 results**

- **4.1 Weather / track-type splits**: `rain` × `circuit_kind` as
  interaction. Data has no weather — would need a supplementary source.
- **4.2 FastF1 telemetry**: lap-level sector times, tyre compounds,
  stint length. Big data dependency, unclear EV before we know what
  fails on 2025 live data.
- **4.3 Hierarchical Bayesian model**: formal prior sharing across
  drivers on the same team. Replaces OpenSkill. Large rewrite; only
  worth it if 3.1–3.3 underperform.
- **4.4 Multi-outcome markets**: H2H matchups, fastest-lap, finishing
  position distribution. MC sampler already produces these for free —
  just needs exposure in the predict CLI.

### Dependency graph

```
    (1.2 data refresh) ───► (1.3 live prediction) ────────────► truly OOS data
                                                                     │
    (1.1 calibration) ──────────────────────────────────────────────┤
                                                                     │
    (2.1 miss pattern)  ──┬─► if A: (3.1 skill drift)               │
                          ├─► if B: (3.2 reliability EWMA)           │
                          └─► mixed: (3.3 quali gap) first           │
                                                                     │
    (2.2 holdout windows) ── informational check on (2.1) ───────────┤
                                                                     ▼
                                                              re-assess Tier 4
```

Fastest path to value: **1.1 → 1.2 → 1.3 → 2.1 → (3.1 or 3.2 or 3.3)**.
Tier 2.2 is a parallel sanity check.

---

## 7. Known issues / non-issues

- **0.6–0.7 overconfidence** is the only known calibration issue.
  Directly addressed by Tier-1 §1.1.
- **Rookie ordinal can be ≤ 0** (e.g., Jack Doohan at Abu Dhabi 2024
  had ordinal ≈ 0). Expected behaviour: μ ≈ 25 − 3·8.33 ≈ 0 for a
  driver with prior-only state. Their p_win ≈ 0 — correctly.
- **Backtest prints duplicate metric lines** when piped through `sed`
  with `,` address ranges. Output format only, doesn't affect data.
- **Top-1 leader baseline > model in 2022–2024**. Tier-2 §2.2 is the
  test that tells us whether this is windowed or structural.
- **No weather, tyre strategy, or telemetry features**. Acknowledged
  v1 simplification; revisited in Tier 4.

---

## 8. Resuming next session

**First thing to do on return**: decide *productionise v1* vs *jump to
modelling upgrades*. The research plan above answers this — Tier 1 is
ordered as it is because without live OOS data we cannot honestly tell
whether any Tier-3 change is real improvement or just noise chasing.

Recommended concrete first step:

```bash
# 1. Sanity-check everything still runs
python scripts/model/backtest.py
python scripts/model/predict_race.py --latest

# 2. Open the decision
#    - If "productionise": start on Tier-1 §1.1 (calibration layer)
#    - If "explore modelling": start on Tier-2 §2.1 (miss-pattern
#      decomposition — it's cheap and tells you which Tier-3 to pick)
```

Key files to re-read before touching anything:
- `src/config.py` — every lever and its rationale.
- `src/model/race_predictor.py` — how features combine, the MC sampler.
- `scripts/model/backtest.py` — walk-forward mechanics, baselines,
  metric definitions.
- This file — the plan.

---

*Last session ended: v1 complete, demo site built, research plan set.*
