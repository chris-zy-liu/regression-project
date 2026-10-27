"""Prospective check: score the frozen Milestone 2 models on releases published after Oct 3, 2026.

Download the current workbook (Census renamed it in Oct 2026) into data/fresh-YYYYMMDD/:
  https://www.census.gov/hfp/btos/downloads/Sector%20by%20Employment%20Size%20Class.xlsx
Run: python3 prospective.py data/fresh-YYYYMMDD/sector_size.xlsx
Forecasts are made exactly as in analysis.py: same groups, model and training rule.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import LAST, MODELS, OUT, SIZES, H, build_pairs, evaluate, load

fresh = Path(sys.argv[1])
latest = max(int(c) for c in pd.read_excel(fresh, "Response Estimates", nrows=0).columns if re.fullmatch(r"20\d{4}", str(c)))
frozen_panel, _, _ = load()
panel, dates, periods = load(fresh, latest)

# Values in the frozen window must be unrevised, and screening must select the same groups.
keys = ["Sector", "Empsize", "period"]
check = frozen_panel.merge(panel, on=keys, suffixes=("", "_new"))
assert len(check) == len(frozen_panel) and np.allclose(check["rate"], check["rate_new"], equal_nan=True)
pairs, groups = build_pairs(panel)
assert groups == build_pairs(frozen_panel)[1]

# New origins: issue releases whose target was published after the frozen snapshot.
origins = [s for s in range(len(periods) - H) if periods[s + H] > LAST]
results, _ = evaluate(panel, pairs, groups, dates, periods, origins)
err = {n: results[n] - results["target"] for n in MODELS}
table = pd.DataFrame({"MAE": {n: e.abs().mean() for n, e in err.items()},
                      "Bias": {n: e.mean() for n, e in err.items()}})
table = table.join(pd.DataFrame({n: e.abs().groupby(results["Empsize"]).mean() for n, e in err.items()}).T
                   .rename(columns=SIZES)).rename(index=MODELS).round(2)
summary = {"fresh_workbook": str(fresh), "latest_release": latest,
           "target_releases": sorted(results["target_period"].unique().tolist()),
           "issue_dates": sorted(results["origin"].dt.date.astype(str).unique().tolist()),
           "forecasts": len(results), "median_target_se_pp": float(results["target_se"].median()),
           "share_regression_errors_within_2se": float((err["regression"].abs() < 2 * results["target_se"]).mean())}
table.to_csv(OUT / "table3_prospective.csv")
results.to_csv(OUT / "prospective_predictions.csv", index=False)
(OUT / "prospective.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
print(table, json.dumps(summary, indent=2, default=str), sep="\n\n")
