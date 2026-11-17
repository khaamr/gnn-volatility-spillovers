"""
Regression checks for the covariance replication protocol.
"""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from statsmodels.stats.correlation_tools import cov_nearest

from evaluation import covariance_losses, long_only_weights, mcs_pvalues, unconstrained_weights
from market_data import PAPER_SYMBOLS, RealizedPanel, realized_measures
from models import (
    Forecasts, correlation_graph_features, forecast_grid, forward_matrices,
    har_lags, make_pd_correlation, normalized_graph, panel_ols,
)


class ReplicationChecks(unittest.TestCase):
    def test_subsampled_covariance_and_quarticity(self):
        slope = np.array([.01, -.013, .018])
        grid = np.arange(391)[None, :, None] * slope
        covariance, quarticity = realized_measures(grid)
        count = np.array([78, 77, 77, 77, 77])
        expected_cov = count.mean() * np.outer(5 * slope, 5 * slope)
        expected_quart = np.mean(count[:, None] ** 2 * (5 * slope) ** 4 / 3, axis=0)
        np.testing.assert_allclose(covariance[0], expected_cov)
        np.testing.assert_allclose(quarticity[0], expected_quart)

    def test_disjoint_lags_and_inclusive_horizons(self):
        levels = np.arange(60, dtype=float)[:, None]
        np.testing.assert_allclose(har_lags(levels)[22, 0], [21, 18.5, 8])
        covariance = np.repeat(np.eye(3)[None], 60, axis=0)
        np.testing.assert_allclose(forward_matrices(covariance, 5), np.repeat((5 * np.eye(3))[None], 56, axis=0))

    def test_line_graph_normalization(self):
        n = 5
        pairs = np.tril_indices(n, -1)
        x = np.arange(30, dtype=float).reshape(1, 10, 3)
        adjacency = np.zeros((10, 10))
        edges = list(zip(*pairs))
        for first, pair in enumerate(edges):
            for second, other in enumerate(edges):
                adjacency[first, second] = first != second and bool(set(pair) & set(other))
        actual = correlation_graph_features(x, pairs, n)["L"][..., 3:]
        np.testing.assert_allclose(actual, normalized_graph(adjacency) @ x)
        self.assertFalse(adjacency.diagonal().any())
        np.testing.assert_equal(adjacency.sum(axis=1), np.full(10, 2 * (n - 2)))

    def test_ols_fixed_effects_with_collinear_features(self):
        rng = np.random.default_rng(10)
        x = rng.normal(size=(60, 4, 3))
        x = np.concatenate([x, x[..., :1]], axis=-1)
        y = x @ np.array([1.2, .4, -.2, 1.2]) + np.arange(4)
        prediction = panel_ols(x, y, np.arange(50), np.arange(50, 60))
        np.testing.assert_allclose(prediction, y[50:], atol=1e-11)

    def test_shared_intercept_matches_pooled_regression(self):
        from sklearn.linear_model import LinearRegression
        rng = np.random.default_rng(7)
        x = rng.normal(size=(60, 4, 3))
        y = rng.normal(size=(60, 4)) + np.arange(4)
        train, test = np.arange(50), np.arange(50, 60)
        expected = LinearRegression().fit(x[train].reshape(-1, 3), y[train].ravel())
        actual = panel_ols(x, y, train, test, intercept_mode="shared")
        np.testing.assert_allclose(
            actual, expected.predict(x[test].reshape(-1, 3)).reshape(10, 4), atol=1e-12,
        )

    def test_statsmodels_projection_preserves_diagonal(self):
        bad = np.array([[1, .99, -.99], [.99, 1, .99], [-.99, .99, 1.]])
        result, count = make_pd_correlation(bad[None].copy())
        self.assertEqual(count, 1)
        np.testing.assert_allclose(result[0], cov_nearest(bad, method="clipped", threshold=1e-4))
        np.testing.assert_allclose(result[0].diagonal(), 1)
        self.assertGreater(np.linalg.eigvalsh(result[0]).min(), 0)

    def test_matrix_loss_formulas(self):
        sigma = np.array([[2., .3], [.3, 1.]])
        actual = np.array([[3., .5], [.5, 2.]])
        forecast = Forecasts(pd.DatetimeIndex(["2024-01-02"]), np.array([0]), actual[None], {"candidate": sigma[None]}, pd.DataFrame())
        row = covariance_losses(forecast).iloc[0]
        self.assertAlmostEqual(row.Euclidean, np.sqrt(2.04))
        self.assertAlmostEqual(row.Frobenius, np.sqrt(2.08))
        self.assertAlmostEqual(row.QLIKE, np.linalg.slogdet(sigma)[1] + np.trace(np.linalg.solve(sigma, actual)))

    def test_portfolio_solutions(self):
        sigma = np.diag([1., 2., 4.])[None]
        expected = np.array([[4/7, 2/7, 1/7]])
        np.testing.assert_allclose(unconstrained_weights(sigma), expected)
        np.testing.assert_allclose(long_only_weights(sigma), expected, atol=1e-6)
        self.assertAlmostEqual(long_only_weights(sigma).sum(), 1)

    def test_mcs_excludes_uniformly_inferior_model(self):
        common = np.arange(40, dtype=float)
        p = mcs_pvalues(pd.DataFrame({"best": common, "worse": common + 1}), draws=200)
        self.assertEqual(p["best"], 1)
        self.assertLess(p["worse"], .05)

    def test_boundary_purging_and_quarticity_choices(self):
        rng = np.random.default_rng(4)
        days = pd.bdate_range("2020-01-02", periods=100)
        n = 3
        lower = rng.normal(scale=.1, size=(len(days), n, n))
        cov = lower @ lower.swapaxes(-1, -2) + np.eye(n)
        panel = RealizedPanel(
            days, PAPER_SYMBOLS[:n], cov, rng.lognormal(size=(len(days), n)),
            rng.normal(size=(len(days), n)), np.zeros((len(days), n)),
            np.ones(len(days)), pd.Series(dtype=int),
        )
        for power in (1., .5):
            results = forecast_grid(panel, horizons=(1, 5), lookback=40, quarticity=True, quarticity_power=power)
            for h, forecast in results.items():
                self.assertTrue((forecast.audit.last_train_target < forecast.audit.first_forecast).all())
                self.assertTrue((forecast.audit.graph_return_through < forecast.audit.first_forecast).all())
                self.assertTrue(forecast.audit.training_labels.eq(40 - h + 1).all())
            self.assertEqual(len(results[1].predictions), 25)
            self.assertTrue(results[1].audit.quarticity_power.eq(power).all())


if __name__ == "__main__":
    unittest.main()
