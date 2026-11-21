"""Regression checks for the generated direct/two-stage comparison."""

from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from two_stage_diagnostics import CATALOG_CUTOFF, design


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'results' / 'two_stage'


class TwoStageChecks(unittest.TestCase):
    def test_unfinished_count_is_missing_and_not_scored(self):
        features = pd.read_csv(OUTPUT / 'stage1_by_issue.csv', parse_dates=['target_reference_end'])
        unfinished = features.target_reference_end > CATALOG_CUTOFF
        self.assertEqual(int(unfinished.sum()), 1)
        self.assertTrue(features.loc[unfinished, 'realized_future_count'].isna().all())
        self.assertFalse(features.loc[unfinished, 'outcome_complete'].any())
        completed = features[~unfinished]
        import json
        summary = json.loads((OUTPUT / 'summary.json').read_text())
        self.assertEqual(summary['stage1']['issue_dates'], len(completed))
        self.assertAlmostEqual(summary['stage1']['poisson_mae'],
                               abs(completed.expected_future_count-completed.realized_future_count).mean())

    def test_all_arms_share_targets_and_direct_reproduces(self):
        predictions = pd.read_csv(OUTPUT / 'predictions.csv')
        self.assertEqual(len(predictions), 706)
        self.assertFalse(predictions[['direct', 'two_stage', 'two_stage_batch', 'oracle']].isna().any().any())
        self.assertAlmostEqual(abs(predictions.direct-predictions.target).mean(), 3.1348793955342336)
        self.assertTrue(np.isfinite(predictions.two_stage_batch).all())

    def test_uncertainty_varies_block_length(self):
        comparisons = pd.read_csv(OUTPUT / 'paired_tests.csv')
        self.assertEqual(set(comparisons.block_length), {2, 3, 4})
        self.assertNotIn('paired_t_p', comparisons.columns)
        self.assertNotIn('sign_test_p', comparisons.columns)
        self.assertTrue((comparisons.ci_low <= comparisons.ci_high).all())

    def test_constant_count_feature_is_finite(self):
        pair = pd.read_csv(ROOT / 'results' / 'forecast_pairs.csv',
                           dtype={'Sector': str, 'Empsize': str})
        train, test = pair.iloc[:120].copy(), pair.iloc[120:150].copy()
        train['expected_batch_count'] = 1.
        test['expected_batch_count'] = 2.
        sectors, sizes = sorted(pair.Sector.unique()), sorted(pair.Empsize.unique())
        a, b = design(train, test, 'two_stage_batch', sectors, sizes)
        self.assertTrue(np.isfinite(a).all())
        self.assertTrue(np.isfinite(b).all())


if __name__ == '__main__':
    unittest.main()
