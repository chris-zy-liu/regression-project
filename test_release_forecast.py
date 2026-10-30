import unittest
import numpy as np
from release_forecast import trend_prediction


class ReleaseForecastTests(unittest.TestCase):
    def test_constant_counts_predict_constant_rate(self):
        self.assertAlmostEqual(trend_prediction(np.full(20, 12.0)), 12.0, places=5)

    def test_increasing_counts_produce_increasing_forecast(self):
        counts = np.arange(4.0, 24.0)
        self.assertGreater(trend_prediction(counts), counts.mean())


if __name__ == '__main__':
    unittest.main()
