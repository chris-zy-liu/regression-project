"""Tests of leakage boundaries, hierarchical selection and data reconstruction."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import extended_analysis as a


class AnalysisTests(unittest.TestCase):
    def test_subset_strong_hierarchy(self):
        indices = {'base': [0, 1]}
        indices.update({name: [i+2] for i, name in enumerate(a.SUBSET_BLOCKS)})
        candidates = a.subset_candidates(indices)
        self.assertGreater(len(candidates), 10)
        for names, columns in candidates:
            self.assertTrue(set(indices['base']).issubset(columns))
            if 'expected_sq' in names or 'rate_expected' in names:
                self.assertIn('expected', names)

    def test_training_imputation_ignores_test(self):
        d = pd.DataFrame({'Sector': ['44']*4, 'Empsize': ['A']*4,
                          'rate': [10., 20., 30., 40.], 'expected': [12., 22., 32., 42.],
                          'step': [0, 1, 2, 3], 'lag1': [np.nan, 10., 20., 30.]})
        # raw() always includes history/cycle block names; absent columns are not
        # selected here, so provide the complete extra schema used by the pipeline.
        extras = ['lag1', 'lag2', 'lag3', 'lag6', 'mean6', 'change1', 'sin6', 'cos6', 'new_sample']
        for col in extras[1:]:
            d[col] = 0.
        f = a.Features(['44'], ['A'], extras)
        x, _ = f.fit_transform(d.iloc[:3], d.iloc[3:])
        altered = d.iloc[3:].copy(); altered['lag1'] = 1000000
        g = a.Features(['44'], ['A'], extras)
        xx, _ = g.fit_transform(d.iloc[:3], altered)
        np.testing.assert_allclose(x, xx)
        np.testing.assert_allclose(f.imputer.statistics_, g.imputer.statistics_)

    def test_nested_splits_respect_horizon(self):
        periods = list(range(202600, 202620))
        pub = pd.date_range('2026-01-01', periods=20, freq='14D')
        dates = pd.DataFrame({'Publication Date': pub}, index=periods)
        rows = [{'step': step, 'target_published': pub[step+3]} for step in range(14) for _ in range(30)]
        folds = a.inner_splits(pd.DataFrame(rows), dates, periods)
        self.assertGreater(len(folds), 0)
        for train, validation in folds:
            step = int(validation.step.iloc[0])
            self.assertLessEqual(train.step.max(), step-3)
            self.assertLessEqual(train.target_published.max(), pub[step])

    def test_rank_and_minimum_norm(self):
        x = np.array([[0., 0.], [1., 1.], [2., 2.]])
        pred, diag = a.least_squares(x, np.array([1., 3., 5.]), x)
        np.testing.assert_allclose(pred, [1, 3, 5], atol=1e-10)
        self.assertEqual(diag['rank'], 2)

    def test_subset_can_recover_curvature(self):
        rng = np.random.default_rng(47)
        def rows(n):
            rate = rng.uniform(2, 70, n)
            r = pd.DataFrame({'Sector': '44', 'Empsize': 'A', 'rate': rate,
                              'expected': rng.uniform(5, 75, n), 'step': rng.integers(0, 15, n)})
            r['target'] = 5 + .2*rate + .015*rate**2 + rng.normal(0, .01, n)
            for c in ['lag1','lag2','lag3','lag6','mean6','change1','sin6','cos6','new_sample']:
                r[c] = 0.
            return r
        extra = ['lag1','lag2','lag3','lag6','mean6','change1','sin6','cos6','new_sample']
        with threadpool_limits(limits=1):
            selected, _, _ = a.subset_choose([(rows(180), rows(80))], ['44'], ['A'], extra)
        self.assertIn('rate_sq', selected)

    def test_extract_reproduces_original_report(self):
        path = Path('data/btos_ai_extract_2026-10-15.csv')
        if not path.exists():
            self.skipTest('Redistributed extract not present')
        panel, dates, periods, _ = a.load_extract(path)
        panel = panel[panel.period <= 202619]
        self.assertEqual(len(panel), 2200)
        self.assertEqual(panel.rate.isna().sum(), 498)
        pairs, groups = a.original.build_pairs(panel)
        self.assertEqual(len(groups), 71)
        self.assertEqual(len(pairs), 1349)
        self.assertEqual(len(pairs.dropna(subset=['rate','expected','target'])), 1335)
        with threadpool_limits(limits=1):
            result, _ = a.original.evaluate(panel, pairs, groups, dates, [p for p in periods if p<=202619])
        self.assertEqual(len(result), 635)
        scores, _ = a.original.scores(result)
        self.assertEqual(scores.loc['Regression (baseline)', 'MAE'], 3.33)
        self.assertEqual(scores.loc['Mean of latest six releases', 'MAE'], 3.40)


if __name__ == '__main__':
    unittest.main()
