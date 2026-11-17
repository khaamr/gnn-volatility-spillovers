"""
Adaptive covariance shapes with delayed exponential-weights updates.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import softmax

from models import BASELINE, GRAPH_MODEL, Forecasts


EXPERTS = (BASELINE, GRAPH_MODEL, "HAR-Cholesky")
ADAPTIVE_MODEL = "Adaptive shape"
UNIFORM_MODEL = "Uniform shape"


@dataclass(frozen=True)
class ShapeMixture:
    forecasts: Forecasts
    weights: pd.DataFrame
    expert_losses: pd.DataFrame
    mixture_loss: pd.Series
    uniform_loss: pd.Series
    regret: pd.Series


def line_graph_spectrum(n_assets):
    """
    Return the eigenvalues and multiplicities of the normalized complete-stock line graph.
    """
    if n_assets < 4:
        raise ValueError("The stated line-graph result requires at least four stocks.")
    return pd.DataFrame({
        "eigenvalue": [1., (n_assets - 4) / (2 * (n_assets - 2)), -1 / (n_assets - 2)],
        "multiplicity": [1, n_assets - 1, n_assets * (n_assets - 3) // 2],
    })


def unit_trace(covariance):
    trace = np.trace(covariance, axis1=-2, axis2=-1)
    if not np.isfinite(covariance).all() or (trace <= 0).any():
        raise ValueError("Expected finite covariance matrices with positive trace.")
    if not np.allclose(covariance, covariance.swapaxes(-1, -2), rtol=1e-9, atol=1e-10):
        raise ValueError("Expected symmetric covariance matrices.")
    shape = covariance / trace[..., None, None]
    if np.min(np.linalg.eigvalsh(shape)) < -1e-10:
        raise ValueError("Expected positive-semidefinite covariance matrices.")
    return shape


def adaptive_shapes(forecast, horizon, learning_rate=.5):
    """
    Update expert weights only after the corresponding covariance targets are observable.
    """
    if horizon < 1 or learning_rate <= 0:
        raise ValueError("Horizon and learning rate must be positive.")
    if not np.all(np.diff(forecast.positions) > 0):
        raise ValueError("Forecast positions must be strictly increasing.")
    for name in EXPERTS:
        np.linalg.cholesky(forecast.predictions[name])
    shapes = np.stack([unit_trace(forecast.predictions[name]) for name in EXPERTS], axis=1)
    realized = unit_trace(forecast.actual)
    losses = .5 * np.sum((shapes - realized[:, None]) ** 2, axis=(-2, -1))
    if losses.min() < -1e-12 or losses.max() > 1 + 1e-12:
        raise ValueError("Shape losses exceeded the trace-one covariance bound.")

    n_dates, n_experts = losses.shape
    weight = np.empty_like(losses)
    cumulative = np.zeros(n_experts)
    next_observation = 0
    unresolved = np.zeros(n_dates, dtype=int)
    for day, origin in enumerate(forecast.positions):
        while (
            next_observation < day
            and forecast.positions[next_observation] + horizon - 1 < origin
        ):
            cumulative += losses[next_observation]
            next_observation += 1
        unresolved[day] = day - next_observation
        weight[day] = softmax(-learning_rate * cumulative)

    mixture = np.sum(weight[..., None, None] * shapes, axis=1)
    uniform = shapes.mean(axis=1)
    trace = np.trace(forecast.predictions[BASELINE], axis1=-2, axis2=-1)
    predictions = {name: forecast.predictions[name] for name in EXPERTS}
    predictions.update({
        f"{name} · common trace": trace[:, None, None] * shapes[:, i]
        for i, name in enumerate(EXPERTS) if name != BASELINE
    })
    predictions[UNIFORM_MODEL] = trace[:, None, None] * uniform
    predictions[ADAPTIVE_MODEL] = trace[:, None, None] * mixture
    mix_loss = .5 * np.sum((mixture - realized) ** 2, axis=(-2, -1))
    uniform_loss = .5 * np.sum((uniform - realized) ** 2, axis=(-2, -1))

    log_k = np.log(n_experts)
    empirical_regret = mix_loss.sum() - losses.sum(axis=0).min()
    delay_cost = learning_rate * unresolved.sum() / 4
    bound = log_k / learning_rate + learning_rate * n_dates / 8 + delay_cost
    general_bound = bound
    if horizon == 1 and learning_rate <= .5:
        bound = log_k / learning_rate
    if empirical_regret > bound + 1e-10:
        raise AssertionError("Adaptive shape regret exceeded the deterministic bound.")
    diagnostic = pd.Series({
        "horizon": horizon,
        "origins": n_dates,
        "learning rate": learning_rate,
        "adaptive cumulative shape loss": mix_loss.sum(),
        "uniform cumulative shape loss": uniform_loss.sum(),
        "best fixed shape loss": losses.sum(axis=0).min(),
        "adaptive regret": empirical_regret,
        "regret upper bound": bound,
        "unresolved-label count": unresolved.sum(),
        "feedback-delay term": delay_cost,
        "general delayed bound": general_bound,
    })
    result = Forecasts(
        forecast.dates, forecast.positions, forecast.actual, predictions, forecast.audit.copy(),
    )
    return ShapeMixture(
        result,
        pd.DataFrame(weight, index=forecast.dates, columns=EXPERTS),
        pd.DataFrame(losses, index=forecast.dates, columns=EXPERTS),
        pd.Series(mix_loss, index=forecast.dates, name=ADAPTIVE_MODEL),
        pd.Series(uniform_loss, index=forecast.dates, name=UNIFORM_MODEL),
        diagnostic,
    )


def loss_risk_counterexample():
    """
    Exhibit smaller covariance losses with strictly higher realized GMVP variance.
    """
    truth = np.diag([1., 4.])
    candidates = {"scaled truth": 2 * truth, "closer forecast": np.diag([1., 3.])}
    rows = []
    for label, matrix in candidates.items():
        error = matrix - truth
        lower = np.tril_indices(2)
        raw = np.linalg.solve(matrix, np.ones(2))
        w = raw / raw.sum()
        rows.append({
            "forecast": label,
            "Euclidean": np.linalg.norm(error[lower]),
            "Frobenius": np.linalg.norm(error),
            "excess QLIKE": np.linalg.slogdet(matrix)[1] + np.trace(np.linalg.solve(matrix, truth))
                            - np.linalg.slogdet(truth)[1] - 2,
            "first weight": w[0],
            "true portfolio variance": w @ truth @ w,
        })
    return pd.DataFrame(rows).set_index("forecast")
