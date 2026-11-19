"""Compare model-level and provider-day batch Poisson forecasts over six 42-day windows."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

from release_process import START, daily_counts, fit_rate, load_catalog


CUTOFF = pd.Timestamp('2026-10-30')
HORIZON = 42
OUTPUT = Path(__file__).resolve().parent / 'results' / 'milestone3'


def compound_pmf(rate, size_probabilities, max_count):
    """Return probabilities for counts 0 through max_count using Poisson recursion."""
    sizes = np.arange(1, len(size_probabilities) + 1)
    weighted = sizes * size_probabilities
    probabilities = np.zeros(max_count + 1)
    probabilities[0] = np.exp(-rate)
    for count in range(1, max_count + 1):
        terms = min(count, len(sizes))
        probabilities[count] = (
            rate * weighted[:terms] @ probabilities[count - sizes[:terms]] / count
        )
    return probabilities


def main():
    catalog = load_catalog(CUTOFF)
    rows = []
    for block in reversed(range(6)):
        end = CUTOFF - pd.Timedelta(days=HORIZON * block)
        start = end - pd.Timedelta(days=HORIZON - 1)
        train_end = start - pd.Timedelta(days=1)
        test_days = pd.date_range(start, end)
        _, counts = daily_counts(catalog, end)
        actual = counts[-HORIZON:].astype(int)

        # A model's date is known in the frozen catalog. Historical catalog
        # availability at each forecast origin cannot be reconstructed.
        past = catalog[catalog.date.between(START, train_end)]
        batch_sizes = past.groupby(['provider', 'date']).size()
        frequencies = np.bincount(batch_sizes)[1:]
        batch_probabilities = frequencies / frequencies.sum()

        for label, is_batch, size_probabilities in (
            ('model_poisson', False, np.array([1.0])),
            ('compound_poisson', True, batch_probabilities),
        ):
            rate = fit_rate(catalog, train_end, 'linear_weekday', batches=is_batch)
            intensity = rate.intensity(test_days)
            expected = intensity.sum() * (np.arange(1, len(size_probabilities) + 1) @ size_probabilities)
            daily = np.vstack([
                compound_pmf(mu, size_probabilities, max(int(actual.max()), 1))
                for mu in intensity
            ])
            total = compound_pmf(intensity.sum(), size_probabilities, max(100, int(actual.sum())))
            cdf = total.cumsum()
            assert cdf[-1] >= .95
            lower, upper = np.searchsorted(cdf, [.05, .95])
            daily_probability = daily[np.arange(HORIZON), actual]
            assert (daily_probability > 0).all() and total[actual.sum()] > 0
            if not is_batch:
                np.testing.assert_allclose(daily_probability, poisson.pmf(actual, intensity))
            rows.append({
                'model': label, 'train_end': train_end.date(), 'start': start.date(),
                'end': end.date(), 'observed': actual.sum(), 'expected': expected,
                'count_absolute_error': abs(actual.sum() - expected),
                'daily_nll': -np.log(daily_probability).mean(),
                'count_nll': -np.log(total[actual.sum()]),
                'lower_90': lower, 'upper_90': upper,
                'covered': lower <= actual.sum() <= upper,
                'expected_multirelease_days': (1 - daily[:, :2].sum(axis=1)).sum(),
                'observed_multirelease_days': (actual >= 2).sum(),
            })

    results = pd.DataFrame(rows)
    summary = results.groupby('model').agg(
        windows=('start', 'size'), count_mae=('count_absolute_error', 'mean'),
        daily_nll=('daily_nll', 'mean'), count_nll=('count_nll', 'mean'),
        windows_covered=('covered', 'sum'),
        expected_multirelease_days=('expected_multirelease_days', 'sum'),
        observed_multirelease_days=('observed_multirelease_days', 'sum'),
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT / 'batch_pilot_windows.csv', index=False)
    summary.to_csv(OUTPUT / 'batch_pilot_summary.csv')
    print(summary.to_string(float_format=lambda value: f'{value:.3f}'))


if __name__ == '__main__':
    main()
