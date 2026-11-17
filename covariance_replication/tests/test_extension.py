"""
Analytic and causal checks for the adaptive covariance extension.
"""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from evaluation import unconstrained_weights
from extension import (
    ADAPTIVE_MODEL, EXPERTS, adaptive_shapes, line_graph_spectrum,
    loss_risk_counterexample, unit_trace,
)
from models import Forecasts


def sample_forecasts(size=35):
    rng = np.random.default_rng(24)
    n = 4
    def matrices():
        a = rng.normal(size=(size, n, n))
        return a @ a.swapaxes(-1, -2) + np.eye(n)
    return Forecasts(
        pd.bdate_range("2024-01-02", periods=size), np.arange(size),
        matrices(), {name: matrices() for name in EXPERTS}, pd.DataFrame(),
    )


class ExtensionChecks(unittest.TestCase):
    def test_line_spectrum_and_contraction(self):
        rng = np.random.default_rng(1)
        for n in (4, 5, 27):
            left, right = np.tril_indices(n, -1)
            incidence = np.zeros((n, len(left)))
            incidence[left, np.arange(len(left))] = 1
            incidence[right, np.arange(len(left))] = 1
            weight = (incidence.T @ incidence - 2 * np.eye(len(left))) / (2 * (n - 2))
            spectrum = line_graph_spectrum(n)
            expected = np.repeat(spectrum.eigenvalue, spectrum.multiplicity)
            np.testing.assert_allclose(np.linalg.eigvalsh(weight), np.sort(expected), atol=1e-12)
            feature = rng.normal(size=(len(left), 3))
            centered = feature - feature.mean(axis=0)
            rate = max(abs(spectrum.eigenvalue.iloc[1:]))
            self.assertLessEqual(np.linalg.norm(weight @ centered), rate * np.linalg.norm(centered) + 1e-12)

    def test_counterexample_improves_losses_and_worsens_risk(self):
        table = loss_risk_counterexample()
        first, second = table.iloc[0], table.iloc[1]
        for metric in ("Euclidean", "Frobenius", "excess QLIKE"):
            self.assertLess(second[metric], first[metric])
        self.assertAlmostEqual(first["true portfolio variance"], .8)
        self.assertAlmostEqual(second["true portfolio variance"], .8125)

    def test_positive_mixture_and_regret(self):
        forecast = sample_forecasts()
        for h in (1, 5, 22):
            result = adaptive_shapes(forecast, horizon=h)
            matrices = result.forecasts.predictions[ADAPTIVE_MODEL]
            self.assertGreater(np.linalg.eigvalsh(matrices).min(), 0)
            np.testing.assert_allclose(
                np.trace(matrices, axis1=-2, axis2=-1),
                np.trace(forecast.predictions[EXPERTS[0]], axis1=-2, axis2=-1),
            )
            self.assertTrue(result.expert_losses.ge(-1e-12).all().all())
            self.assertTrue(result.expert_losses.le(1 + 1e-12).all().all())
            self.assertTrue((result.mixture_loss <= (result.weights * result.expert_losses).sum(axis=1) + 1e-12).all())
            self.assertLessEqual(result.regret["adaptive regret"], result.regret["regret upper bound"] + 1e-12)
            if h == 1:
                self.assertAlmostEqual(result.regret["regret upper bound"], 2 * np.log(3))

    def test_future_outcomes_do_not_change_earlier_weights(self):
        forecast = sample_forecasts()
        changed = forecast.actual.copy()
        changed[0] = forecast.predictions[EXPERTS[1]][0]
        altered = Forecasts(
            forecast.dates, forecast.positions, changed, forecast.predictions, forecast.audit,
        )
        for h in (1, 5):
            original = adaptive_shapes(forecast, h)
            revision = adaptive_shapes(altered, h)
            np.testing.assert_allclose(original.weights.iloc[:h], revision.weights.iloc[:h])
            self.assertGreater(np.max(np.abs(original.weights.iloc[h] - revision.weights.iloc[h])), 1e-8)

    def test_relative_spectral_portfolio_bound(self):
        truth = np.diag([1., 2., 4., 8.])
        forecast = truth.copy()
        forecast[0, 0] *= 1.2
        forecast[3, 3] *= .8
        oracle = unconstrained_weights(truth[None])[0]
        candidate = unconstrained_weights(forecast[None])[0]
        actual_risk = candidate @ truth @ candidate
        oracle_risk = oracle @ truth @ oracle
        self.assertLessEqual(actual_risk, 1.2 / .8 * oracle_risk + 1e-12)

    def test_shape_validation_rejects_indefinite_matrix(self):
        with self.assertRaises(ValueError):
            unit_trace(np.array([[[1., 2.], [2., 1.]]]))


if __name__ == "__main__":
    unittest.main()
