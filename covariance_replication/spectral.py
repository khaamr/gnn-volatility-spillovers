"""
Geometric covariance pooling with a causal regime gate and spectral cleaning.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from extension import ADAPTIVE_MODEL, EXPERTS, UNIFORM_MODEL, adaptive_shapes
from models import Forecasts


SPECTRAL_MODEL = "Geometric spectral gate"
STATIC_MODEL = "Static geometric pool"
BANK_NAMES = (*EXPERTS, "Arithmetic HAR covariance", "Log-Euclidean HAR covariance")
DTYPE = torch.float64


@dataclass(frozen=True)
class SpectralResult:
    forecasts: Forecasts
    gate_weights: pd.DataFrame
    audit: pd.DataFrame
    bank: dict[str, np.ndarray]


def symmetric_function(matrices, function):
    values, vectors = np.linalg.eigh(matrices)
    return (vectors * function(values)[..., None, :]) @ vectors.swapaxes(-1, -2)


def expert_bank(panel, forecast):
    """
    Construct all additional experts and gate inputs from strictly earlier sessions.
    """
    covariance = panel.covariance
    n = covariance.shape[-1]
    mean_variance = np.trace(covariance, axis1=-2, axis2=-1) / n
    stable = covariance + 1e-8 * mean_variance[:, None, None] * np.eye(n)
    logarithm = symmetric_function(stable, np.log)
    prefix = np.concatenate([np.zeros_like(covariance[:1]), np.cumsum(covariance, axis=0)])
    log_prefix = np.concatenate([np.zeros_like(logarithm[:1]), np.cumsum(logarithm, axis=0)])
    positions = forecast.positions
    if (positions < 22).any():
        raise ValueError("Every spectral forecast needs 22 earlier covariance observations.")
    daily = stable[positions - 1]
    weekly = (prefix[positions] - prefix[positions - 5]) / 5
    monthly = (prefix[positions] - prefix[positions - 22]) / 22
    mean_log = (
        logarithm[positions - 1]
        + (log_prefix[positions] - log_prefix[positions - 5]) / 5
        + (log_prefix[positions] - log_prefix[positions - 22]) / 22
    ) / 3
    arithmetic = (daily + weekly + monthly) / 3
    geometric = symmetric_function(mean_log, np.exp)
    bank = np.stack([
        *(forecast.predictions[name] for name in EXPERTS), arithmetic, geometric,
    ], axis=1)
    np.linalg.cholesky(bank)

    trace = np.trace(bank, axis1=-2, axis2=-1)
    base_values = np.linalg.eigvalsh(bank[:, 0])
    diagonal = np.diagonal(daily, axis1=-2, axis2=-1)
    std = np.sqrt(diagonal)
    correlation = daily / (std[:, :, None] * std[:, None, :])
    mean_correlation = (correlation.sum(axis=(1, 2)) - n) / (n * (n - 1))
    features = np.column_stack([
        np.log(trace / trace[:, :1]),
        np.log(base_values[:, -1] / base_values[:, 0]),
        np.log(np.trace(daily, axis1=-2, axis2=-1) / np.trace(monthly, axis1=-2, axis2=-1)),
        np.log(np.trace(weekly, axis1=-2, axis2=-1) / np.trace(monthly, axis1=-2, axis2=-1)),
        np.std(np.log(diagonal), axis=1),
        mean_correlation,
    ])
    if not np.isfinite(features).all():
        raise ValueError("The geometric expert bank produced nonfinite gate features.")
    return bank, features


def project_simplex(values):
    """
    Euclidean projection of each row onto nonnegative weights summing to one.
    """
    ordered = values.sort(dim=-1, descending=True).values
    index = torch.arange(1, values.shape[-1] + 1, dtype=values.dtype, device=values.device)
    threshold = (ordered.cumsum(dim=-1) - 1) / index
    active = (ordered > threshold).sum(dim=-1, keepdim=True) - 1
    offset = threshold.gather(-1, active)
    return torch.clamp(values - offset, min=0)


def differentiable_weights(covariance, steps=40):
    """
    Approximate long-only minimum variance using accelerated projected gradients.
    """
    n = covariance.shape[-1]
    weights = torch.full_like(covariance[..., 0], 1 / n)
    extrapolated = weights
    largest = torch.linalg.eigvalsh(covariance)[..., -1:].detach()
    momentum = 1.
    for _ in range(steps):
        gradient = (covariance @ extrapolated.unsqueeze(-1)).squeeze(-1)
        updated = project_simplex(extrapolated - gradient / largest)
        next_momentum = (1 + np.sqrt(1 + 4 * momentum ** 2)) / 2
        extrapolated = updated + ((momentum - 1) / next_momentum) * (updated - weights)
        weights, momentum = updated, next_momentum
    return weights


def condition_spectrum(values, limit):
    """
    Shrink toward the mean eigenvalue just enough to enforce a condition bound.
    """
    mean = values.mean(dim=-1, keepdim=True)
    excess = torch.clamp(values.max(dim=-1, keepdim=True).values
                         - limit * values.min(dim=-1, keepdim=True).values, min=0)
    shrink = excess / (excess + (limit - 1) * mean)
    return (1 - shrink) * values + shrink * mean


class GeometricSpectralGate(nn.Module):
    def __init__(self, n_features, n_experts, hidden=8, condition_limit=100, dynamic=True):
        super().__init__()
        if condition_limit <= 1:
            raise ValueError("The condition limit must exceed one.")
        self.n_experts = n_experts
        self.condition_limit = condition_limit
        self.dynamic = dynamic
        self.register_buffer("location", torch.zeros(n_features, dtype=DTYPE))
        self.register_buffer("scale", torch.ones(n_features, dtype=DTYPE))
        if dynamic:
            self.gate = nn.Sequential(
                nn.Linear(n_features, hidden), nn.Tanh(), nn.Linear(hidden, n_experts + 1),
            )
            self.cleaner = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)
            nn.init.zeros_(self.cleaner[-1].weight)
            nn.init.constant_(self.cleaner[-1].bias, -3.)
        else:
            self.logits = nn.Parameter(torch.zeros(n_experts + 1, dtype=DTYPE))
            self.shrink_logit = nn.Parameter(torch.tensor(-3., dtype=DTYPE))
        self.double()

    def standardize(self, features):
        with torch.no_grad():
            self.location.copy_(features.mean(dim=0))
            self.scale.copy_(features.std(dim=0, unbiased=False).clamp_min(1e-6))

    def forward(self, bank, features):
        if self.dynamic:
            transformed = ((features - self.location) / self.scale).clamp(-8, 8)
            output = self.gate(transformed)
        else:
            output = self.logits.expand(len(bank), -1)
        probability = output[:, :self.n_experts].softmax(dim=-1)
        pooled = (bank * probability[..., None, None]).sum(dim=1)
        eigenvalues, eigenvectors = torch.linalg.eigh(pooled)
        mean = eigenvalues.mean(dim=-1, keepdim=True)
        if self.dynamic:
            relative = torch.log(eigenvalues / mean)
            condition = torch.log(eigenvalues[:, -1:] / eigenvalues[:, :1])
            spectral_input = torch.stack([relative, condition.expand_as(relative)], dim=-1)
            shrink = self.cleaner(spectral_input).squeeze(-1).sigmoid()
        else:
            shrink = self.shrink_logit.sigmoid()
        cleaned = (1 - shrink) * eigenvalues + shrink * mean
        cleaned = cleaned * (eigenvalues.sum(dim=-1, keepdim=True) / cleaned.sum(dim=-1, keepdim=True))
        cleaned = condition_spectrum(cleaned, self.condition_limit)
        variance_scale = torch.exp(.25 * torch.tanh(output[:, -1]))
        covariance = (eigenvectors * cleaned[:, None, :]) @ eigenvectors.transpose(-1, -2)
        return covariance * variance_scale[:, None, None], probability


def objective_parts(predicted, actual, returns, steps):
    distance = torch.linalg.matrix_norm(predicted - actual, ord="fro", dim=(-2, -1)).mean()
    weights = differentiable_weights(predicted, steps)
    mean_square_return = (weights * returns).sum(dim=-1).square().mean()
    return torch.stack([distance, mean_square_return])


def train_model(bank, features, actual, returns, config, dynamic, epochs, reference, monitor=None):
    """
    Train one fixed seed; optional monitoring uses only a later historical block.
    """
    torch.manual_seed(config["seed"])
    model = GeometricSpectralGate(
        features.shape[1], bank.shape[1], config["hidden_units"],
        config["condition_limit"], dynamic,
    )
    model.standardize(features)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )
    checkpoints = {}
    for epoch in range(epochs + 1):
        if monitor is not None and epoch in config["epoch_checkpoints"]:
            with torch.no_grad():
                validation_bank, validation_x, validation_actual, validation_returns, denominator = monitor
                estimate, _ = model(validation_bank, validation_x)
                parts = objective_parts(
                    estimate, validation_actual, validation_returns,
                    config["differentiable_simplex_steps"],
                )
                checkpoints[epoch] = (parts / denominator).numpy()
        if epoch == epochs:
            break
        optimizer.zero_grad()
        predicted, _ = model(bank, features)
        parts = objective_parts(
            predicted, actual, returns, config["differentiable_simplex_steps"],
        )
        loss = .5 * (parts / reference).sum()
        if not torch.isfinite(loss):
            raise RuntimeError("The joint spectral training loss became nonfinite.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
    return model, checkpoints


def spectral_forecasts(panel, base, config, progress=None):
    """
    Refit on fixed historical folds and retain every comparison model.
    """
    bank, features = expert_bank(panel, base)
    tensors = [torch.as_tensor(x, dtype=DTYPE) for x in (
        bank, features, base.actual, panel.portfolio_return[base.positions],
    )]
    bank_t, features_t, actual_t, returns_t = tensors
    before = adaptive_shapes(base, 1)
    periods = base.dates.to_period(config["gate_refit_period"])
    output = {SPECTRAL_MODEL: [], STATIC_MODEL: []}
    selected_positions, gates, audit = [], [], []
    eligible = (np.arange(len(base.dates)) >= config["minimum_history"]) & (
        base.dates >= config["evaluation_start"]
    )
    for period in periods[eligible].unique():
        test = np.flatnonzero((periods == period) & eligible)
        origin = test[0]
        train = np.arange(max(0, origin - config["rolling_history"]), origin)
        validation = train[-config["inner_validation"]:]
        fitting = train[:-config["inner_validation"]]
        if len(fitting) < 252 or base.positions[train[-1]] >= base.positions[origin]:
            raise ValueError("The spectral fold failed its historical split.")
        with torch.no_grad():
            fit_reference = objective_parts(bank_t[fitting, 0], actual_t[fitting], returns_t[fitting],
                                            config["differentiable_simplex_steps"])
            validation_reference = objective_parts(
                bank_t[validation, 0], actual_t[validation], returns_t[validation],
                config["differentiable_simplex_steps"],
            )
            full_reference = objective_parts(bank_t[train, 0], actual_t[train], returns_t[train],
                                             config["differentiable_simplex_steps"])
        for name, dynamic in ((SPECTRAL_MODEL, True), (STATIC_MODEL, False)):
            _, checkpoints = train_model(
                bank_t[fitting], features_t[fitting], actual_t[fitting], returns_t[fitting],
                config, dynamic, max(config["epoch_checkpoints"]), fit_reference,
                monitor=(bank_t[validation], features_t[validation], actual_t[validation],
                         returns_t[validation], validation_reference),
            )
            chosen_epoch = min(checkpoints, key=lambda epoch: (max(checkpoints[epoch]), epoch))
            model, _ = train_model(
                bank_t[train], features_t[train], actual_t[train], returns_t[train],
                config, dynamic, chosen_epoch, full_reference,
            )
            with torch.no_grad():
                predicted, probabilities = model(bank_t[test], features_t[test])
            output[name].append(predicted.numpy())
            if dynamic:
                gates.append(probabilities.numpy())
            audit.append({
                "period": str(period), "model": name, "first_forecast": base.dates[origin],
                "last_fit_label": base.dates[fitting[-1]],
                "first_validation_label": base.dates[validation[0]],
                "last_validation_label": base.dates[validation[-1]],
                "last_refit_label": base.dates[train[-1]],
                "historical_labels": len(train), "chosen_epoch": chosen_epoch,
                "validation_frobenius_ratio": checkpoints[chosen_epoch][0],
                "validation_risk_proxy_ratio": checkpoints[chosen_epoch][1],
            })
        selected_positions.append(test)
        if progress is not None:
            progress(f"{period}: historical labels {len(train)}; forecasts {len(test)}")
    if not selected_positions:
        raise ValueError("No complete historical folds for the spectral extension.")
    selected = np.concatenate(selected_positions)
    predictions = {name: values[selected] for name, values in base.predictions.items() if name in EXPERTS}
    predictions.update({name: before.forecasts.predictions[name][selected]
                        for name in (UNIFORM_MODEL, ADAPTIVE_MODEL)})
    predictions.update({name: bank[selected, i] for i, name in enumerate(BANK_NAMES) if name not in EXPERTS})
    predictions.update({name: np.concatenate(values) for name, values in output.items()})
    audit_frame = pd.DataFrame(audit)
    result = Forecasts(
        base.dates[selected], base.positions[selected], base.actual[selected],
        predictions, audit_frame,
    )
    return SpectralResult(
        result, pd.DataFrame(np.concatenate(gates), index=result.dates, columns=BANK_NAMES),
        audit_frame, {name: bank[:, i] for i, name in enumerate(BANK_NAMES)},
    )
