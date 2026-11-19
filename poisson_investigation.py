"""Nested temporal investigation of compound Poisson release forecasts.

Run from the repository root with python poisson_investigation.py.
The fixed October 30 catalog and all previously examined windows are retrospective.
"""

from functools import lru_cache
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import poisson
import statsmodels.api as sm
from threadpoolctl import threadpool_limits

from milestone3_batch_pilot import compound_pmf
from release_process import START, load_catalog

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'poisson_investigation'
CUTOFF = pd.Timestamp('2026-10-30')
HORIZON = 42
SPECS = {
    'constant': ('constant', None, 'unit'),
    'trend': ('trend', None, 'unit'),
    'weekday': ('weekday', None, 'unit'),
    'batch_constant': ('constant', None, 'pooled'),
    'batch_trend': ('trend', None, 'pooled'),
    'batch_weekday': ('weekday', None, 'pooled'),
    'batch_day_of_week': ('day_of_week', None, 'pooled'),
    'batch_quadratic': ('quadratic', None, 'pooled'),
    'batch_recent_365': ('weekday', 365, 'pooled'),
    'batch_recent_504': ('weekday', 504, 'pooled'),
    'batch_recent_sizes': ('weekday', None, 'recent'),
    'batch_provider_sizes': ('weekday', None, 'provider'),
}


def design(days, form):
    days = pd.DatetimeIndex(days)
    time = (days - START).days.to_numpy() / 365.25
    parts = [np.ones(len(days))]
    if form != 'constant':
        parts.append(time)
    if form == 'quadratic':
        parts.append(time ** 2)
    if form in ('weekday', 'quadratic'):
        parts.append((days.dayofweek < 5).astype(float))
    if form == 'day_of_week':
        parts.extend((days.dayofweek == day).astype(float) for day in range(1, 7))
    return np.column_stack(parts)


def size_probabilities(batches, cutoff, method):
    if method == 'unit':
        return np.array([1.])
    sizes = batches['size'].to_numpy()
    global_probability = np.bincount(sizes)[1:].astype(float)
    global_probability /= global_probability.sum()
    recent = batches[batches.date > cutoff - pd.Timedelta(days=180)]
    if method == 'recent':
        counts = np.bincount(recent['size'], minlength=len(global_probability) + 1)[1:]
        return (counts + 20 * global_probability) / (len(recent) + 20)
    if method == 'provider':
        # Shrink each provider's mark distribution and recent batch share.
        # Provider frequencies forecast the mixture of sizes, not causal behavior.
        probability = np.zeros_like(global_probability)
        shares = batches.groupby('provider').size()
        recent_shares = recent.groupby('provider').size()
        for provider, group in batches.groupby('provider'):
            counts = np.bincount(group['size'], minlength=len(probability) + 1)[1:]
            conditional = (counts + 10 * global_probability) / (len(group) + 10)
            share = (recent_shares.get(provider, 0) + 20 * shares[provider] / len(batches)) / (len(recent) + 20)
            probability += share * conditional
        np.testing.assert_allclose(probability.sum(), 1.)
        return probability
    return global_probability


def make_forecaster(catalog):
    @lru_cache(maxsize=None)
    def forecast(name, cutoff):
        cutoff = pd.Timestamp(cutoff)
        form, window, method = SPECS[name]
        past = catalog[catalog.date.between(START, cutoff)]
        batches = past.groupby(['provider', 'date']).size().rename('size').reset_index()
        start = max(START, cutoff - pd.Timedelta(days=window - 1)) if window else START
        days = pd.date_range(start, cutoff)
        events = past if method == 'unit' else batches
        y = events.groupby('date').size().reindex(days, fill_value=0).to_numpy()
        x = design(days, form)
        fit = sm.GLM(y, x, family=sm.families.Poisson()).fit(maxiter=150)
        future = pd.date_range(cutoff + pd.Timedelta(days=1), periods=HORIZON)
        intensity = np.exp(np.clip(design(future, form) @ fit.params, -25, 10))
        weights = size_probabilities(batches, cutoff, method)
        return intensity, weights
    return forecast


def score(catalog, forecast, name, cutoff):
    intensity, weights = forecast(name, cutoff)
    days = pd.date_range(cutoff + pd.Timedelta(days=1), periods=HORIZON)
    observed = catalog.groupby('date').size().reindex(days, fill_value=0).to_numpy(int)
    mean = intensity.sum() * (np.arange(1, len(weights) + 1) @ weights)
    support = max(200, int(mean + 20 * np.sqrt(mean + 1)), int(observed.sum()))
    total = compound_pmf(intensity.sum(), weights, support)
    assert total.sum() > .999
    daily = np.vstack([compound_pmf(rate, weights, max(1, int(observed.max()))) for rate in intensity])
    probabilities = daily[np.arange(HORIZON), observed]
    if SPECS[name][2] == 'unit':
        np.testing.assert_allclose(probabilities, poisson.pmf(observed, intensity))
    lower, upper = np.searchsorted(total.cumsum(), [.05, .95])
    value = int(observed.sum())
    interval_score = upper - lower + 20 * max(lower - value, 0) + 20 * max(value - upper, 0)
    return dict(model=name, train_end=cutoff, test_start=days[0], test_end=days[-1],
                observed=value, expected=mean, absolute_error=abs(value - mean),
                count_nll=-np.log(max(total[value], 1e-300)),
                daily_nll=-np.log(np.maximum(probabilities, 1e-300)).mean(),
                lower=lower, upper=upper, covered=lower <= value <= upper,
                width=upper - lower, interval_score=interval_score,
                expected_busy_days=(1 - daily[:, :2].sum(axis=1)).sum(),
                observed_busy_days=int((observed >= 2).sum()))


def main():
    catalog = load_catalog(CUTOFF)
    forecast = make_forecaster(catalog)
    OUT.mkdir(parents=True, exist_ok=True)
    outer, validation, selected = [], [], []
    candidates = [name for name in SPECS if name.startswith('batch_')]
    for block in reversed(range(6)):
        cutoff = CUTOFF - pd.Timedelta(days=HORIZON * (block + 1))
        inner = []
        for earlier in range(4):
            inner_cutoff = cutoff - pd.Timedelta(days=HORIZON * (earlier + 1))
            for name in candidates:
                row = score(catalog, forecast, name, inner_cutoff)
                assert row['test_end'] <= cutoff
                inner.append(row)
                validation.append(dict(outer_cutoff=cutoff, **row))
        losses = pd.DataFrame(inner).groupby('model').count_nll.mean()
        chosen = losses.idxmin()
        for name in SPECS:
            outer.append(score(catalog, forecast, name, cutoff))
        chosen_row = score(catalog, forecast, chosen, cutoff)
        selected.append(dict(chosen=chosen, validation_nll=losses[chosen], **chosen_row))
        outer.append(dict(chosen_row, model='nested_selection'))
        print(f'{cutoff.date()}: selected {chosen}', flush=True)
    windows = pd.DataFrame(outer)
    summary = windows.groupby('model').agg(
        windows=('observed', 'size'), observed_total=('observed', 'sum'),
        count_mae=('absolute_error', 'mean'), count_nll=('count_nll', 'mean'),
        daily_nll=('daily_nll', 'mean'), covered=('covered', 'sum'),
        mean_width=('width', 'mean'), interval_score=('interval_score', 'mean'),
        expected_busy_days=('expected_busy_days', 'sum'), observed_busy_days=('observed_busy_days', 'sum'))
    windows.to_csv(OUT / 'windows.csv', index=False)
    summary.to_csv(OUT / 'summary.csv')
    pd.DataFrame(validation).to_csv(OUT / 'validation.csv', index=False)
    pd.DataFrame(selected).to_csv(OUT / 'selected.csv', index=False)
    errors = windows.pivot(index='test_end', columns='model', values='absolute_error')
    comparisons = []
    for name in summary.index:
        delta = errors[name] - errors.weekday
        comparisons.append(dict(model=name, mean_error_difference=delta.mean(),
                                windows_better=int((delta < 0).sum()),
                                best_window_difference=delta.min(), worst_window_difference=delta.max()))
    pd.DataFrame(comparisons).to_csv(OUT / 'paired.csv', index=False)
    final_scores = pd.DataFrame([score(catalog, forecast, name, CUTOFF - pd.Timedelta(days=42 * k))
                                 for name in candidates for k in range(1, 5)])
    final_model = final_scores.groupby('model').count_nll.mean().idxmin()
    rate, weights = forecast(final_model, CUTOFF)
    final_probability = compound_pmf(rate.sum(), weights, 300)
    final_mean = rate.sum() * (np.arange(1, len(weights) + 1) @ weights)
    final = dict(snapshot=str(CUTOFF.date()), selected=final_model, forecast_start='2026-10-31',
                 forecast_end='2026-12-11', expected_count=final_mean,
                 interval90=np.searchsorted(final_probability.cumsum(), [.05, .95]).tolist(),
                 interval_type='conditional on fitted rate and empirical sizes',
                 evaluation='retrospective fixed catalog, six previously examined windows')
    (OUT / 'forecast.json').write_text(json.dumps(final, indent=2) + '\n')
    visible = ['constant', 'weekday', 'batch_weekday', 'nested_selection']
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    for name in visible:
        group = windows[windows.model == name]
        axes[0].plot(group.test_end, group.expected, 'o-', label=name.replace('_', ' '))
    observed = windows[windows.model == 'constant']
    axes[0].plot(observed.test_end, observed.observed, 'k--', label='Observed')
    axes[0].set_ylabel('Models per 42 days')
    axes[0].tick_params(axis='x', rotation=40)
    axes[0].legend(fontsize=6)
    show = summary.loc[visible]
    axes[1].barh([s.replace('_', ' ') for s in visible], show.interval_score)
    axes[1].set_xlabel('90% interval score (lower is better)')
    fig.tight_layout()
    fig.savefig(OUT / 'comparison.pdf', bbox_inches='tight')
    fig.savefig(OUT / 'comparison.png', dpi=160, bbox_inches='tight')
    print(summary.round(3).to_string())
    print(json.dumps(final, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
