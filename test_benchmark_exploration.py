import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class BenchmarkExplorationTests(unittest.TestCase):
    def test_forecasts_are_bounded_and_target_dates_follow_issue_dates(self):
        with tempfile.TemporaryDirectory() as temp:
            subprocess.run([sys.executable, 'benchmark_exploration.py',
                            'data/epoch-benchmark-data-2026-10-29.zip', '--output', temp],
                           check=True, capture_output=True)
            result = pd.read_csv(Path(temp) / 'benchmark_backtest.csv')
        self.assertGreater(len(result), 0)
        self.assertTrue((pd.to_datetime(result.target_date) > pd.to_datetime(result.issue)).all())
        for column in ['observed_best', 'carry', 'trend']:
            self.assertTrue(result[column].between(0, 1).all())


if __name__ == '__main__':
    unittest.main()
