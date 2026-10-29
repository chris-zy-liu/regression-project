"""Retrospective GPQA score-trend feasibility check, not a public-availability backtest."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path, default=Path('benchmark_exploration'))
    args = parser.parse_args()
    with ZipFile(args.archive) as z:
        data = pd.read_csv(z.open('gpqa_diamond.csv'))
    start = pd.to_datetime(data['Started at'], utc=True, errors='coerce')
    release = pd.to_datetime(data['Release date'], utc=True, errors='coerce')
    score = pd.to_numeric(data['mean_score'], errors='coerce')
    valid = start.notna() & release.notna() & score.between(0, 1)
    # A run cannot be known before either the model release or the run start.
    # This lower bound is not a verified public-availability date.
    ready = pd.concat([start[valid], release[valid]], axis=1).max(axis=1)
    events = pd.DataFrame({'earliest_possible': ready, 'score': score[valid]})
    months = pd.date_range('2025-02-01', '2026-10-01', freq='MS', tz='UTC')
    best = []
    for end in months:
        previous = events.loc[events.earliest_possible < end, 'score']
        best.append(previous.max() if len(previous) else np.nan)
    series = pd.Series(best, index=months).dropna()
    rows = []
    for i in range(12, len(series) - 2):
        train = series.iloc[:i + 1].to_numpy()
        recent = train[-6:]
        slope, intercept = np.polyfit(np.arange(len(recent)), recent, 1)
        rows.append({'issue': series.index[i].date().isoformat(),
                     'target_date': series.index[i + 2].date().isoformat(),
                     'observed_best': float(series.iloc[i + 2]),
                     'carry': float(train[-1]),
                     'trend': float(np.clip(intercept + slope * (len(recent) + 1), 0, 1))})
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError('Not enough historical months for this exploratory comparison')
    args.output.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output / 'benchmark_backtest.csv', index=False)
    series.rename('best_score').rename_axis('as_of').to_csv(args.output / 'benchmark_series.csv')
    summary = {'archive_sha256': hashlib.sha256(args.archive.read_bytes()).hexdigest(),
               'source': 'https://epoch.ai/benchmarks/use-this-data',
               'benchmark': 'GPQA Diamond, Epoch internal runs, mean_score',
               'comparison_count': len(result),
               'carry_mae_score_points': float((result.carry - result.observed_best).abs().mean() * 100),
               'trend_mae_score_points': float((result.trend - result.observed_best).abs().mean() * 100),
               'limitations': 'Retrospective current catalogue; run start and model release are only a lower bound on public availability. Scores can be revised and models have different evaluation setups. Selection, time range and rolling window were chosen after observing the dataset. No causal or prospective claim.'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
