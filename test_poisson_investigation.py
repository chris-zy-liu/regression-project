"""Independent checks of count probabilities and historical filtering."""

import unittest

import numpy as np
import pandas as pd
from scipy.stats import poisson

from poisson_investigation import compound_pmf, load_catalog, make_forecaster, size_probabilities


class CompoundPoissonChecks(unittest.TestCase):
    def test_unit_batches_equal_poisson(self):
        values = compound_pmf(4., np.array([1.]), 80)
        np.testing.assert_allclose(values, poisson.pmf(np.arange(81), 4.), atol=1e-14)

    def test_batch_moments_and_normalization(self):
        values = compound_pmf(3., np.array([.5, .5]), 100)
        counts = np.arange(len(values))
        mean = counts @ values
        self.assertAlmostEqual(values.sum(), 1.)
        self.assertAlmostEqual(mean, 4.5)
        self.assertAlmostEqual((counts - mean) ** 2 @ values, 7.5)

    def test_future_events_do_not_change_forecasts(self):
        catalog = load_catalog(pd.Timestamp('2026-10-30'))
        cutoff = pd.Timestamp('2026-02-20')
        past = catalog[catalog.date <= cutoff]
        full_forecast = make_forecaster(catalog)
        past_forecast = make_forecaster(past)
        for name in ['batch_weekday', 'batch_provider_sizes', 'batch_recent_365']:
            a = full_forecast(name, cutoff)
            b = past_forecast(name, cutoff)
            np.testing.assert_allclose(a[0], b[0])
            np.testing.assert_allclose(a[1], b[1])

    def test_provider_probabilities_normalize(self):
        frame = pd.DataFrame({'provider': ['A', 'A', 'B'],
                              'date': pd.to_datetime(['2025-01-01', '2025-08-01', '2025-08-02']),
                              'size': [1, 2, 3]})
        probability = size_probabilities(frame, pd.Timestamp('2025-08-03'), 'provider')
        self.assertAlmostEqual(probability.sum(), 1.)
        self.assertTrue((probability >= 0).all())


if __name__ == '__main__':
    unittest.main()
