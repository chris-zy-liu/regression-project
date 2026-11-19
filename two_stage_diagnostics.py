"""Why does the two-stage model not beat direct forecasting? (Milestone 3 feedback)

python two_stage_diagnostics.py

Runs after release_process.py. Uses the same 71 groups, 10 issue dates, 3-release
horizon, expanding window and inner ridge tuning as release_process.adoption_evaluate.

Arms (all ridge, same tuning grid):
  direct     : control features only (current rate, lags, industry, size, trend ...)
  two_stage  : direct + Poisson-predicted release count for the target window (stage 1)
  oracle     : direct + ACTUAL release count in the target window (stage 1 made perfect;
               uses future information, so it is a diagnostic, not a usable forecast)

Comparing oracle with direct asks "is release information useful at all?".
Comparing two_stage with oracle asks "how much do stage-1 errors cost?".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from threadpoolctl import threadpool_limits

import extended_analysis as previous
import release_process as rp

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/two_stage'
SEED = 403130
ARMS = ['direct', 'two_stage', 'oracle']
GRID = [.1, 1, 10, 100, 1000, 10000, 100000]


def realized_counts(catalog, features):
    """Actual catalog releases in (issue, issue+24 days], the window stage 1 forecasts."""
    out = []
    for _, r in features.iterrows():
        issue, end = pd.Timestamp(r.issue), pd.Timestamp(r.target_reference_end)
        out.append(int(((catalog.date > issue) & (catalog.date <= end)).sum()))
    return np.array(out)


def design(train, test, arm, sectors, sizes):
    f = previous.Features(sectors, sizes, rp.EXTRA)
    xa, xb = f.fit_transform(train, test)
    if arm == 'direct':
        return xa, xb
    col = 'expected_future_count' if arm == 'two_stage' else 'realized_future_count'
    m, s = train[col].mean(), train[col].std() or 1.
    return (np.column_stack([xa, (train[col]-m)/s]), np.column_stack([xb, (test[col]-m)/s]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = rp.load_catalog('2026-10-30')
    features = pd.read_csv(ROOT/'results/releases/issue_features.csv', parse_dates=['issue', 'target_reference_end'])
    features['realized_future_count'] = realized_counts(catalog, features)
    features['stage1_error'] = features.expected_future_count-features.realized_future_count

    # ---------- Stage 1: how good is the Poisson forecast of releases? ----------
    rng = np.random.default_rng(SEED)
    lam = features.expected_future_count.to_numpy()
    # Irreducible error: even if lambda were exactly right, Poisson counts scatter around it.
    floor = np.array([np.abs(rng.poisson(l, 200000)-l).mean() for l in lam])
    # Naive stage-1 baselines, both known on the issue date.
    prior24 = np.array([int(((catalog.date > i-pd.Timedelta(days=24)) & (catalog.date <= i)).sum())
                        for i in features.issue])
    hist_rate = np.array([((catalog.date >= rp.START) & (catalog.date <= i)).sum()/max((i-rp.START).days, 1)*24
                          for i in features.issue])
    y = features.realized_future_count.to_numpy()
    stage1 = {
        'issue_dates': len(features),
        'mean_actual_count_24d': float(y.mean()),
        'poisson_mae': float(np.abs(lam-y).mean()),
        'carry_forward_last24_mae': float(np.abs(prior24-y).mean()),
        'historical_average_rate_mae': float(np.abs(hist_rate-y).mean()),
        'poisson_noise_floor_mae': float(floor.mean()),
        'poisson_bias': float((lam-y).mean()),
        'skill_vs_carry_forward': float(1-np.abs(lam-y).mean()/np.abs(prior24-y).mean()),
        'skill_vs_historical_average': float(1-np.abs(lam-y).mean()/np.abs(hist_rate-y).mean()),
    }
    # 90% Poisson interval coverage of the actual counts.
    lo, hi = stats.poisson.ppf(.05, lam), stats.poisson.ppf(.95, lam)
    stage1['coverage_90'] = float(((y >= lo) & (y <= hi)).mean())
    diag = pd.read_csv(ROOT/'results/releases/rate_diagnostics.csv').set_index('model')
    stage1['pearson_dispersion_selected'] = float(diag.loc['linear_weekday', 'pearson_dispersion'])
    stage1['pearson_dispersion_linear'] = float(diag.loc['linear', 'pearson_dispersion'])
    cmp = pd.read_csv(ROOT/'results/releases/rate_model_comparison.csv').set_index('model')
    stage1['rate_cv_42day_mae'] = cmp.mean_42day_abs_error.to_dict()

    # ---------- Is the stage-1 feature anything more than a time trend? ----------
    t = np.arange(len(features))
    stage1_signal = {
        'distinct_values_per_issue_date': 'one value shared by all 71 groups',
        'corr_expected_count_with_time': float(np.corrcoef(features.expected_future_count, t)[0, 1]),
        'corr_actual_count_with_time': float(np.corrcoef(features.realized_future_count, t)[0, 1]),
        'corr_expected_with_actual': float(np.corrcoef(features.expected_future_count, features.realized_future_count)[0, 1]),
    }
    r2 = stage1_signal['corr_expected_count_with_time']**2
    stage1_signal['vif_expected_count_on_time'] = float(1/(1-r2))
    features.to_csv(OUT/'stage1_by_issue.csv', index=False)

    # ---------- Stage 2: direct vs two-stage vs oracle ----------
    pair = pd.read_csv(ROOT/'results/forecast_pairs.csv', dtype={'Sector': str, 'Empsize': str},
                       parse_dates=['published', 'target_published'])
    calendar = pd.read_csv(ROOT/'results/release_calendar.csv', parse_dates=['Publication Date'])
    dates, periods = calendar.set_index('period'), calendar.period.tolist()
    pair = pair.merge(features[['period', 'expected_future_count', 'realized_future_count']],
                      on='period', validate='many_to_one')
    sectors, sizes = sorted(pair.Sector.unique()), sorted(pair.Empsize.unique())
    preds, fit_rows = [], []
    for step in sorted(pair[pair.step >= 10].step.unique()):
        issue = dates.loc[periods[int(step)], 'Publication Date']
        train = pair[pair.target_published <= issue]
        test = pair[pair.step == step].copy()
        assert train.target_published.max() <= issue < test.target_published.min()
        folds = previous.inner_splits(train, dates, periods)
        for arm in ARMS:
            losses = []
            for a, b in folds:
                xa, xb = design(a, b, arm, sectors, sizes)
                losses.append([np.abs(previous.ridge_predict(xa, a.target, xb, v)-b.target).mean() for v in GRID])
            v = GRID[int(np.argmin(np.mean(losses, axis=0)))]
            xa, xb = design(train, test, arm, sectors, sizes)
            test[arm] = previous.ridge_predict(xa, train.target, xb, v)
            insample = np.abs(previous.ridge_predict(xa, train.target, xa, v)-train.target).mean()
            fit_rows.append({'origin': issue, 'arm': arm, 'alpha': v, 'train_mae': insample,
                             'test_mae': np.abs(test[arm]-test.target).mean()})
        test['carry_forward'], test['six_release_mean'] = test.rate, test.mean6
        test['origin'] = issue
        preds.append(test)
        print(f'origin {issue.date()} done', flush=True)
    res = pd.concat(preds, ignore_index=True)
    res.to_csv(OUT/'predictions.csv', index=False)
    fits = pd.DataFrame(fit_rows)
    fits.to_csv(OUT/'fit_by_origin.csv', index=False)

    models = ['carry_forward', 'six_release_mean'] + ARMS
    perf = []
    for m in models:
        e = res[m]-res.target
        row = {'model': m, 'n': len(res), 'mae': e.abs().mean(), 'rmse': float(np.sqrt((e**2).mean())), 'bias': e.mean()}
        for size, d in res.groupby('Empsize'):
            row[f'mae_{size}'] = (d[m]-d.target).abs().mean()
        perf.append(row)
    perf = pd.DataFrame(perf)
    perf.to_csv(OUT/'performance.csv', index=False)
    by_origin = pd.DataFrame({m: (res[m]-res.target).abs().groupby(res.origin).mean() for m in models})
    by_origin.to_csv(OUT/'mae_by_origin.csv')

    tests = []
    for m, ref in [('two_stage', 'direct'), ('oracle', 'direct'), ('two_stage', 'oracle'), ('direct', 'six_release_mean')]:
        d = (by_origin[m]-by_origin[ref]).to_numpy()
        tt = stats.ttest_1samp(d, 0.)
        sign = stats.binomtest(int((d < 0).sum()), len(d))
        tests.append({'model': m, 'reference': ref, 'mean_mae_difference': d.mean(), 'wins': int((d < 0).sum()),
                      'origins': len(d), 'paired_t_p': tt.pvalue, 'sign_test_p': sign.pvalue})
    tests = pd.DataFrame(tests)
    tests.to_csv(OUT/'paired_tests.csv', index=False)

    gap = fits.groupby('arm')[['train_mae', 'test_mae']].mean()
    gap['gap'] = gap.test_mae-gap.train_mae
    gap.to_csv(OUT/'overfitting_gap.csv')
    within = float((np.abs(res.direct-res.target) <= 2*res.target_se).mean())

    summary = {'stage1': stage1, 'stage1_signal': stage1_signal,
               'stage2_mae': perf.set_index('model').mae.to_dict(),
               'paired_tests': tests.to_dict('records'),
               'overfitting_gap': gap.to_dict('index'),
               'direct_errors_within_2se': within}
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2, default=float)+'\n')
    print(perf.round(3).to_string(index=False))
    print(tests.round(3).to_string(index=False))
    print(gap.round(3))
    print(json.dumps({'stage1': stage1, 'stage1_signal': stage1_signal, 'within2se': within}, indent=2, default=float))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
