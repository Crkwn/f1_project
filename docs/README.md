# F1 Win-Probability Model — demo site

This directory is a **self-contained static site** you can serve on
GitHub Pages. It reads the back-test output from `reports/backtest/`
(converted to a single `data.json`) and renders it as an interactive
demo.

## Preview it locally

```bash
python3 -m http.server 8765 --directory docs
# then open http://localhost:8765
```

Any HTTP server works — GitHub Pages, Netlify, S3, etc. No build step
and no framework; it's one HTML file, one CSS file, one JS file, one
JSON file.

## Regenerate after a new back-test run

```bash
# 1. Re-run the back-test (writes fresh predictions + metrics)
python scripts/model/backtest.py

# 2. Rebuild this site's data.json from the new artefacts
python scripts/web/build_site.py
```

The site will be live at
   `https://crkwn.github.io/<repo-name>/`.

## What the site shows

- Headline stats from the walk-forward holdout: average probability
  assigned to the actual winner, top-1 and top-3 hit rates, average
  rank of the winner.
- Per-race ranked prediction table — P(win), P(podium), P(points),
  plus fair odds and 20%-overround posted odds.
- Calibration diagram — do the 70% predictions actually happen 70%
  of the time?
- Bucket hit rate — "when the model says 60–70% on its top pick,
  how often is it right?"
- Model-vs-baselines comparison (grid-softmax, pole-empirical,
  last-race-winner, championship-leader).

The data reflects the v1 model's back-test over 2022–2024 using
2014–2021 as the training window.
