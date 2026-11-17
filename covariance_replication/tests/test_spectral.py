"""
Structural and causal checks for the geometric spectral architecture.
"""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from market_data import RealizedPanel
from models import Forecasts
from spectral import (
    DTYPE, EXPERTS, GeometricSpectralGate, condition_spectrum,
    differentiable_weights, expert_bank, objective_parts, project_simplex,
)


class SpectralChecks(unittest.TestCase):
    def test_condition_guard_preserves_trace_and_bounds_condition(self):
        values = torch.tensor([[1e-8, 1., 100.], [1., 2., 3.]], dtype=DTYPE)
        cleaned = condition_spectrum(values, 100)
        self.assertTrue(torch.all(cleaned > 0))
        torch.testing.assert_close(cleaned.sum(dim=1), values.sum(dim=1))
        self.assertLessEqual(float((cleaned.max(dim=1).values / cleaned.min(dim=1).values).max()), 100 + 1e-9)
        torch.testing.assert_close(cleaned[1], values[1])

    def test_simplex_projection_and_diagonal_portfolio(self):
        value = torch.tensor([[.2, -.5, 1.2], [.3, .4, .5]], dtype=DTYPE)
        projected = project_simplex(value)
        self.assertTrue(torch.all(projected >= 0))
        torch.testing.assert_close(projected.sum(dim=1), torch.ones(2, dtype=DTYPE))
        covariance = torch.diag(torch.tensor([1., 2., 4.], dtype=DTYPE))[None]
        weights = differentiable_weights(covariance, steps=40)
        torch.testing.assert_close(weights[0], torch.tensor([4/7, 2/7, 1/7], dtype=DTYPE), atol=1e-5, rtol=1e-5)

    def test_network_is_positive_conditioned_and_permutation_equivariant(self):
        torch.manual_seed(7)
        factor = torch.randn(5, 5, 4, 4, dtype=DTYPE)
        bank = factor @ factor.transpose(-1, -2) + .001 * torch.eye(4, dtype=DTYPE)
        features = torch.randn(5, 10, dtype=DTYPE)
        model = GeometricSpectralGate(10, 5)
        permutation = torch.tensor([2, 0, 3, 1])
        for dynamic in (True, False):
            model = GeometricSpectralGate(10, 5, dynamic=dynamic)
            estimate, probability = model(bank, features)
            eig = torch.linalg.eigvalsh(estimate)
            self.assertTrue(torch.all(eig > 0))
            self.assertLessEqual(float((eig[:, -1] / eig[:, 0]).max().detach()), 100 + 1e-8)
            torch.testing.assert_close(probability.sum(dim=1), torch.ones(5, dtype=DTYPE))
            permuted, _ = model(bank[:, :, permutation][:, :, :, permutation], features)
            torch.testing.assert_close(
                permuted, estimate[:, permutation][:, :, permutation], atol=1e-9, rtol=1e-9,
            )

    def test_joint_training_has_finite_gradients(self):
        torch.manual_seed(2)
        factor = torch.randn(6, 5, 4, 4, dtype=DTYPE)
        bank = factor @ factor.transpose(-1, -2) + torch.eye(4, dtype=DTYPE)
        features = torch.randn(6, 10, dtype=DTYPE)
        model = GeometricSpectralGate(10, 5)
        estimate, _ = model(bank, features)
        returns = torch.randn(6, 4, dtype=DTYPE) * .01
        losses = objective_parts(estimate, bank[:, 0], returns, 40)
        (losses / torch.tensor([10., .0001], dtype=DTYPE)).mean().backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_experts_do_not_read_future_realized_covariance(self):
        rng = np.random.default_rng(3)
        factor = rng.normal(size=(70, 4, 4))
        matrices = factor @ factor.swapaxes(-1, -2) + np.eye(4)
        dates = pd.bdate_range("2024-01-02", periods=70)
        positions = np.arange(22, 70)
        forecast = Forecasts(
            dates[positions], positions, matrices[positions],
            {name: matrices[positions - 1].copy() for name in EXPERTS}, pd.DataFrame(),
        )
        def panel(covariance):
            return RealizedPanel(
                dates, ("a", "b", "c", "d"), covariance, np.ones((70, 4)),
                np.ones((70, 4)), np.ones((70, 4)) * .01, np.ones(70), pd.Series(dtype=float),
            )
        original_bank, original_features = expert_bank(panel(matrices), forecast)
        changed = matrices.copy()
        changed[50:] *= 10
        revised_bank, revised_features = expert_bank(panel(changed), forecast)
        available = positions <= 50
        np.testing.assert_allclose(original_bank[available], revised_bank[available])
        np.testing.assert_allclose(original_features[available], revised_features[available])
        self.assertGreater(np.max(np.abs(original_bank[~available] - revised_bank[~available])), 1)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
