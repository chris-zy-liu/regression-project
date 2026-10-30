"""Exploratory one-month forecasts of Epoch's catalogued language-model counts.

This is a retrospective catalogue analysis, not an as-of-vintage backtest.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


def trend_prediction(counts):
    t = np.arange(len(counts), dtype=float)
    def objective(beta):
        eta = np.clip(beta[0] + beta[1] * t, -20, 20)
        return np.sum(np.exp(eta) - counts * eta)
    def gradient(beta):
        residual = np.exp(np.clip(beta[0] + beta[1] * t, -20, 20)) - counts
        return np.array([residual.sum(), residual @ t])
    fit = minimize(objective, [np.log(max(np.mean(counts), .01)), 0.0], jac=gradient, method='BFGS')
    if not fit.success and np.linalg.norm(fit.jac) > 1e-4:
        raise RuntimeError(f'Poisson fit failed: {fit.message}')
    return float(np.exp(np.clip(fit.x[0] + fit.x[1] * len(counts), -20, 20)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('catalogue', type=Path)
    parser.add_argument('--output', type=Path, default=Path('release_exploration'))
    args = parser.parse_args()
    raw = pd.read_csv(args.catalogue, low_memory=False)
    date = pd.to_datetime(raw['Publication date'], errors='coerce')
    language = raw.Domain.fillna('').str.contains('Language', case=False, regex=False)
    selected = raw[language & date.between('2024-01-01', '2026-09-30')].copy()
    selected['month'] = date.loc[selected.index].dt.to_period('M')
    months = pd.period_range('2024-01', '2026-09', freq='M')
    counts = selected.groupby('month').size().reindex(months, fill_value=0)
    forecasts = []
    for i in range(12, len(counts)):
        training = counts.iloc[:i].to_numpy(dtype=float)
        forecasts.append({'month': str(months[i]), 'observed': int(counts.iloc[i]),
                          'constant_mean': float(training.mean()),
                          'trend_mean': trend_prediction(training)})
    out = pd.DataFrame(forecasts)
    final_mean = trend_prediction(counts.to_numpy(dtype=float))
    args.output.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output / 'monthly_backtest.csv', index=False)
    counts.rename('count').rename_axis('month').to_csv(args.output / 'monthly_counts.csv')
    summary = {'source_sha256': hashlib.sha256(args.catalogue.read_bytes()).hexdigest(),
               'source': 'https://epoch.ai/data/ai-models',
               'selection': 'Epoch catalog rows with Language in Domain, publication date 2024-01 through 2026-09',
               'models': len(selected), 'months': len(counts), 'backtest_months': len(out),
               'constant_mae': float((out.constant_mean - out.observed).abs().mean()),
               'trend_mae': float((out.trend_mean - out.observed).abs().mean()),
               'october_2026_trend_expected_count': final_mean,
               'october_2026_poisson_95pct_interval': [int(x) for x in poisson.interval(.95, final_mean)],
               'limitations': 'Retrospective catalogue; publication date is not necessarily product release. Selection, reporting delays, revisions, non-independence and overdispersion are untested. Interval excludes parameter and catalogue uncertainty. Not a causal adoption predictor.'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
