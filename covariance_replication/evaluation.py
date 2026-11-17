"""
Covariance losses, model confidence sets, and minimum-variance portfolios.
"""

import numpy as np
import cvxpy as cp
import pandas as pd


METRICS = ("Euclidean", "Frobenius", "QLIKE")


def covariance_losses(forecast):
    """
    Score the lower triangle including the diagonal, the full matrix, and Gaussian QLIKE.
    """
    n_assets = forecast.actual.shape[-1]
    lower = np.tril_indices(n_assets)
    frames = []
    for model, predicted in forecast.predictions.items():
        delta = forecast.actual - predicted
        sign, logdet = np.linalg.slogdet(predicted)
        if not np.all(sign > 0) or not np.isfinite(predicted).all():
            raise ValueError(f"Invalid covariance predictions for {model}.")
        np.linalg.cholesky(predicted)
        solved = np.linalg.solve(predicted, forecast.actual)
        frames.append(pd.DataFrame({
            "date": forecast.dates, "model": model,
            "Euclidean": np.linalg.norm(delta[:, lower[0], lower[1]], axis=1),
            "Frobenius": np.linalg.norm(delta, axis=(1, 2)),
            "QLIKE": logdet + np.trace(solved, axis1=-2, axis2=-1),
        }))
    return pd.concat(frames, ignore_index=True)


def point_ratios(losses, baseline="HAR-DRD"):
    level = losses.groupby("model", sort=False)[list(METRICS)].mean()
    denominator = level.loc[baseline].where(level.loc[baseline] > 0)
    return level.div(denominator)


def bootstrap_means(losses, draws, block, rng):
    """
    Resample dates jointly using circular stationary bootstrap blocks.
    """
    n_days, n_models = losses.shape
    result = np.empty((draws, n_models))
    for first in range(0, draws, 128):
        last = min(first + 128, draws)
        positions = rng.integers(0, n_days, size=(n_days, last - first))
        restart = rng.random(positions.shape) < 1 / block
        for day in range(1, n_days):
            positions[day] = np.where(restart[day], positions[day], (positions[day - 1] + 1) % n_days)
        result[first:last] = losses[positions].mean(axis=0)
    return result


def mcs_pvalues(daily_loss, draws=10_000, block=2, seed=4274989):
    """
    Compute the sequential SQ model confidence set with stationary bootstrap.
    """
    data = daily_loss.to_numpy()
    expected = data.mean(axis=0)
    centered = bootstrap_means(data, draws, block, np.random.default_rng(seed)) - expected
    alive = list(range(data.shape[1]))
    output = {}
    previous_p = 0.0

    def studentize(value, sims):
        sd = np.sqrt(np.mean(sims ** 2, axis=0))
        fallback = np.where(value > 0, np.inf, np.where(value < 0, -np.inf, 0))
        return np.divide(value, sd, out=fallback, where=sd > 0), np.divide(
            sims, sd, out=np.zeros_like(sims), where=sd > 0
        )

    while len(alive) > 1:
        a, b = np.triu_indices(len(alive), 1)
        subset = np.array(alive)
        i, j = subset[a], subset[b]
        z, bootstrap = studentize(expected[i] - expected[j], centered[:, i] - centered[:, j])
        simulated = np.sum(bootstrap ** 2, axis=1)
        p_value = (1 + np.count_nonzero(simulated >= np.sum(z ** 2))) / (draws + 1)
        previous_p = max(previous_p, p_value)
        sample = expected[subset]
        panel = centered[:, subset]
        score, _ = studentize(sample - sample.mean(), panel - panel.mean(axis=1, keepdims=True))
        removed = alive.pop(int(np.argmax(score)))
        output[daily_loss.columns[removed]] = previous_p
    output[daily_loss.columns[alive[0]]] = 1.0
    return pd.Series(output).reindex(daily_loss.columns)


def forecast_table(losses, baseline="HAR-DRD", draws=10_000, block=2):
    ratios = point_ratios(losses, baseline)
    columns = {}
    for metric in METRICS:
        daily = losses.pivot(index="date", columns="model", values=metric).reindex(columns=ratios.index)
        columns[(metric, "ratio")] = ratios[metric]
        columns[(metric, "rank")] = daily.mean().rank(method="min").astype(int)
        columns[(metric, "MCS p")] = mcs_pvalues(daily, draws=draws, block=block)
    return pd.DataFrame(columns)


def unconstrained_weights(covariance):
    one = np.ones((*covariance.shape[:-1], 1))
    raw = np.linalg.solve(covariance, one)[..., 0]
    return raw / raw.sum(axis=1, keepdims=True)


def long_only_weights(covariance):
    """
    Minimize forecast variance on the simplex using a reusable sparse QP.
    """
    n = covariance.shape[-1]
    weights = cp.Variable(n)
    matrix = cp.Parameter((n, n), symmetric=True)
    objective = cp.Minimize(cp.quad_form(weights, cp.psd_wrap(matrix)))
    problem = cp.Problem(objective, [cp.sum(weights) == 1, weights >= 0])
    result = np.empty((len(covariance), n))
    for index, estimate in enumerate(covariance):
        matrix.value = estimate / np.mean(np.diag(estimate))
        problem.solve(
            solver=cp.OSQP, warm_start=True, ignore_dpp=True, verbose=False,
            eps_abs=1e-7, eps_rel=1e-7,
        )
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise ValueError(f"Long-only solve failed: {problem.status}")
        value = weights.value
        if value is None or np.min(value) < -1e-5 or abs(value.sum() - 1) > 1e-5:
            raise ValueError("Long-only solution failed the simplex constraint.")
        value = np.maximum(value, 0)
        result[index] = value / value.sum()
    return result


def portfolio_stats(weights, returns, horizon=1):
    """
    Compute realized annualized risk and drift-adjusted turnover.
    """
    portfolio = np.sum(weights * returns, axis=1)
    if (1 + portfolio <= 0).any():
        return np.nan, np.nan
    post = weights[:-1] * (1 + returns[:-1]) / (1 + portfolio[:-1, None])
    risk = 100 * np.std(portfolio, ddof=0) * np.sqrt(252 / horizon)
    turnover = np.abs(weights[1:] - post).sum(axis=1).mean()
    return risk, turnover


def portfolio_table(forecast, daily_returns, horizon=1, selected=None):
    """
    Rebalance each of the nonoverlapping horizon cohorts and average their diagnostics.
    """
    prefix = np.vstack([np.zeros(daily_returns.shape[1]), np.cumsum(np.log1p(daily_returns), axis=0)])
    starts = forecast.positions
    realized = np.expm1(prefix[starts + horizon] - prefix[starts])
    names = forecast.predictions if selected is None else selected
    weights = {"1/N": np.full(realized.shape, 1 / realized.shape[1])}
    for model in names:
        weights[(model, "GMVP")] = unconstrained_weights(forecast.predictions[model])
        weights[(model, "GMVP+")] = long_only_weights(forecast.predictions[model])
    records = []
    for key, w in weights.items():
        name, kind = ("1/N", "equal") if key == "1/N" else key
        stats = [portfolio_stats(w[k::horizon], realized[k::horizon], horizon)
                 for k in range(horizon) if len(w[k::horizon]) > 1]
        records.append((name, kind, *np.mean(stats, axis=0)))
    return pd.DataFrame(records, columns=["model", "portfolio", "annual risk (%)", "turnover"])
