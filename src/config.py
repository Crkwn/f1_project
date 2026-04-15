from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_CACHE = ROOT / "data" / "cache"

# --- API ---
JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
API_DELAY_SECONDS = 1.0          # be polite to the free API

# --- Data range ---
HISTORY_START_YEAR = 2000        # how far back to pull history
# (pre-2000 F1 is a very different sport — cars, rules, reliability all changed massively)

# --- Elo model hyperparameters (legacy, kept for comparison benchmark) ---
ELO_INITIAL = 1500               # starting rating for every new driver
ELO_K = 32                       # update step size per pairwise comparison
ELO_SCALE = 400                  # controls how rating differences map to win probabilities

# --- Score model ---
MIN_RACES_FOR_RELIABLE_SCORE = 5  # below this, treat score with lower confidence

# ---------------------------------------------------------------------------
# Stage 1a rating model (OpenSkill-PlackettLuce wrapped with F1 corrections)
# ---------------------------------------------------------------------------

# Scope: when does the rating history begin? 2014 picks up the hybrid-era
# cutoff — the cleanest modern rule-package break. Pre-2014 history is
# used as descriptive EDA context only; it does NOT feed the rating.
RATING_START_YEAR = 2014

# OpenSkill's default priors — neutral μ = 25, σ = 25/3 ≈ 8.33. These are
# unitless; the absolute scale is arbitrary as long as we're consistent.
RATING_INITIAL_MU    = 25.0
RATING_INITIAL_SIGMA = 25.0 / 3.0

# Qualifying as a secondary weekend update: qualifying outcomes are a
# cleaner skill signal than race outcomes (no strategy, no safety cars,
# no tyre management). We include quali as its own update per weekend,
# slightly up-weighted vs the race. 1.0 = same weight as race; we start
# at 1.0 in v1 and tune from back-testing.
QUALI_UPDATE_WEIGHT = 1.0

# Structural σ inflations — model the fact that our uncertainty about a
# driver's skill widens in specific observable situations.
#
#   * team switch   → new-car fit is unobserved; widen σ at transition
#   * inactivity    → skill may have drifted during the gap (EDA Q16:
#                     gaps ≥3 yrs showed a meaningful return penalty)
#
# These are additive bumps in σ-squared space (variance adds).
SIGMA_BUMP_TEAM_SWITCH        = 1.5   # one-time bump at team change
SIGMA_BUMP_PER_YEAR_INACTIVE  = 0.7   # per full year missed
SIGMA_FLOOR                   = 1.0   # never let σ shrink below this

# Event-filter threshold for tail outliers (post-hoc safety net). After a
# full rating update, if a race's residual z-score for the driver exceeds
# this we recognise the race as an "event" (damage, wet chaos, etc.) and
# down-weight. In practice this is a belt-and-braces check ON TOP of the
# DNF-family weights, not a substitute for them.
EVENT_RESIDUAL_Z_CAP          = 4.0

# ---------------------------------------------------------------------------
# Stage 2 scoring — how we convert Stage 1a (μ, σ) into a feature for the
# race-level win-probability model.
# ---------------------------------------------------------------------------
#
# Decision (v1): use ORDINAL (μ − 3σ) as the driver-skill feature, then
# softmax over the field with a single fitted temperature τ.
#
# Why not per-race z-score of μ?
#   Z-scoring has three negatives we want to avoid:
#     (a) Field-composition sensitivity — a weak driver on the grid
#         inflates the field std and compresses every other driver's z.
#         P(win) should move with skill and car, not with who else showed up.
#     (b) Softmax-amplification — exp() turns small z shifts from (a) into
#         noticeable probability shifts.
#     (c) It throws away σ — a driver with few races (wide σ) should be
#         treated more cautiously than an established one at the same μ.
#
# Why ordinal + τ?
#   - Ordinal = μ − 3σ is field-invariant (property of the driver, not
#     the grid) and already penalises wide-σ drivers.
#   - A single fitted temperature τ absorbs the "absolute scale drifted"
#     problem that z-scoring was originally meant to fix. We tune τ from
#     back-test log-loss rather than assume a scale.
#
# This is a LEVER. If the back-test reveals ordinal + τ miscalibrates in
# a way z-scoring would have caught, flip DRIVER_SCORE_MODE and re-run.
# The alternatives recognised by Stage 2:
#   "ordinal"  — μ − 3σ        (current default)
#   "mu"       — raw μ         (ignores σ; simplest)
#   "z_mu"     — per-race z of μ            (field-relative)
#   "z_ordinal"— per-race z of ordinal      (field-relative, σ-aware)
DRIVER_SCORE_MODE = "ordinal"

# Softmax temperature. Larger τ = flatter probabilities; smaller τ = sharper.
#
# Fitted value τ = 20 from 2022-2024 back-test: log-loss minimised at
# 1.40 (vs. 2.10 for the grid-position baseline). My initial intuition
# (τ ≈ 80 based on "one-third of feature span") was about 4× too flat —
# the empirical optimum is considerably sharper than the scale argument
# suggested, because combined_score spans across a 20-driver field
# interact with PL's implicit normalisation in ways the back-of-envelope
# misses. Keep this as a lever: if we retune features, refit τ.
SOFTMAX_TAU_INIT = 20.0

# Stage 2 feature weights — initial values from domain knowledge / EDA.
# The ordinal feature is the dominant signal (skill gap is the biggest
# driver-level differentiator); pace (prev-year constructor rank) and
# grid position are secondary. These are deliberately left as named
# hyperparameters so back-test can refit them.
#
# Scaling: ordinal is already on ~[100,300] absolute. pace_score and
# grid_score are normalised to [0,1], so their weights are the
# "absolute contribution range" — w_pace=50 means the best-vs-worst
# gap contributed by pace is 50 points, comparable to ~1/4 of the skill
# range. Grid at w_grid=30 reflects that grid position matters (pole →
# win ~40% in modern F1) but less than career skill over the long run.
W_DRIVER_ORDINAL = 1.0    # ordinal is already on a "units of μ" scale
W_PACE_PREV_RANK = 50.0   # pace_score ∈ [0, 1]
W_GRID           = 30.0   # grid_score ∈ [0, 1]

# Monte-Carlo sample count for PL-draw-based outputs (P(podium), P(points)).
# A few thousand is plenty: MC std for a 10% probability at n=5000 is
# ~0.4pp, well below any calibration tolerance we'd care about.
MC_SAMPLES = 5000
