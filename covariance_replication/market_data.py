"""
Market inputs and synchronized realized covariance construction.
"""

from dataclasses import dataclass
from pathlib import Path
from shutil import which
from subprocess import run
from urllib.request import urlretrieve

import numpy as np
import pandas as pd


SECTORS = {
    "AAPL": "technology", "MSFT": "technology", "INTC": "technology",
    "CSCO": "technology", "CRM": "technology", "IBM": "technology",
    "DIS": "communication", "VZ": "communication",
    "HD": "discretionary", "MCD": "discretionary", "NKE": "discretionary",
    "PG": "staples", "KO": "staples", "WMT": "staples",
    "JNJ": "health", "UNH": "health", "MRK": "health", "AMGN": "health",
    "JPM": "finance", "AXP": "finance", "GS": "finance", "TRV": "finance",
    "BA": "industrials", "HON": "industrials", "MMM": "industrials",
    "CAT": "industrials", "CVX": "energy",
}
PAPER_SYMBOLS = tuple(SECTORS)
DATASET = "https://huggingface.co/datasets/syncster-data/historical-market-data"
REVISION = "eff0dac55d843b3b83caec819efa213021a3e3e5"
OPEN_MINUTE = 9 * 60 + 30
MINUTES = 390


@dataclass(frozen=True)
class RealizedPanel:
    dates: pd.DatetimeIndex
    symbols: tuple[str, ...]
    covariance: np.ndarray
    quarticity: np.ndarray
    graph_return: np.ndarray
    portfolio_return: np.ndarray
    spy_variance: np.ndarray
    session_counts: pd.Series


def download_ticker(ticker, directory):
    """
    Cache one public Parquet file at a fixed dataset revision.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{ticker}.parquet"
    if path.exists():
        return path
    partial = path.with_suffix(".parquet.part")
    url = f"{DATASET}/resolve/{REVISION}/data/stocks/1m/{ticker[0]}/{ticker}.parquet"
    try:
        if curl := which("curl"):
            run([curl, "-fSL", "--retry", "2", "--max-time", "150", url, "-o", str(partial)], check=True)
        else:
            urlretrieve(url, partial)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return path


def minute_panel(ticker, source, directory, start, end):
    """
    Align to the New York session grid without filling prices across sessions.
    """
    if source == "syncster":
        path = download_ticker(ticker, directory)
        bars = pd.read_parquet(path, columns=["ts", "open", "close"])
    elif source == "lobster":
        bars = pd.read_csv(directory / f"{ticker}.csv", usecols=["ts", "open", "close"])
    else:
        raise ValueError("Choose syncster or lobster minute files.")

    stamp = pd.to_datetime(bars.pop("ts"), utc=True).dt.tz_convert("America/New_York")
    minute = (stamp.dt.hour * 60 + stamp.dt.minute) - OPEN_MINUTE
    day = stamp.dt.tz_localize(None).dt.normalize()
    keep = day.between(start, end) & minute.between(0, MINUTES - 1) & (day.dt.dayofweek < 5)
    bars = bars.loc[keep].assign(date=day[keep], minute=minute[keep])
    if bars.duplicated(["date", "minute"]).any():
        raise ValueError(f"Duplicate regular-session minute bars for {ticker}.")
    close = bars.pivot(index="date", columns="minute", values="close").reindex(columns=range(MINUTES))
    close = close.where(np.isfinite(close) & close.gt(0))
    opening = bars.loc[bars.minute.eq(0)].set_index("date")["open"].reindex(close.index)
    enough = close.count(axis=1).ge(350) & opening.gt(0)
    filled = close.ffill(axis=1, limit=10)
    valid = enough & filled.notna().all(axis=1) & np.isfinite(opening)
    prices = np.column_stack([opening.loc[valid], filled.loc[valid]])
    return pd.DataFrame(prices, index=close.index[valid])


def realized_measures(log_prices):
    """
    Average five return-grid covariance and quarticity estimates for each day.
    """
    if log_prices.ndim != 3 or log_prices.shape[1] != MINUTES + 1:
        raise ValueError("Expected a day × 391 grid points × assets array.")
    n_days, _, n_assets = log_prices.shape
    covariance = np.zeros((n_days, n_assets, n_assets))
    quarticity = np.zeros((n_days, n_assets))
    for offset in range(5):
        moves = np.diff(log_prices[:, offset::5, :], axis=1)
        covariance += moves.swapaxes(1, 2) @ moves / 5
        quarticity += (moves.shape[1] / 15) * np.sum(moves ** 4, axis=1)
    return covariance, quarticity


def load_market(source, directory, start, end, symbols=PAPER_SYMBOLS):
    """
    Build daily covariance, quarticity, and returns on a shared session calendar.
    """
    if len(symbols) < 3 or len(set(symbols)) != len(symbols) or "SPY" in symbols:
        raise ValueError("Provide at least three distinct stock tickers, excluding SPY.")
    tickers = (*symbols, "SPY")
    frames = {ticker: minute_panel(ticker, source, directory, start, end) for ticker in tickers}
    days = frames["SPY"].index
    for ticker in symbols:
        days = days.intersection(frames[ticker].index)
    days = days.sort_values()
    if not len(days):
        raise ValueError("No shared, sufficiently complete regular trading sessions.")
    prices = np.stack([frames[ticker].loc[days].to_numpy() for ticker in tickers], axis=-1)
    log_prices = np.log(prices) * 100
    covariance, quarticity = realized_measures(log_prices[:, :, :-1])
    spy_cov, _ = realized_measures(log_prices[:, :, -1:])
    fwd_return = prices[:, -1, :-1] / prices[:, 0, :-1] - 1
    graph_return = log_prices[:, -1, :-1] - log_prices[:, 0, :-1]
    diag = np.diagonal(covariance, axis1=-2, axis2=-1)
    valid = np.isfinite(covariance).all(axis=(1, 2)) & (diag > 0).all(axis=1)
    if not valid.all():
        raise ValueError("The aligned panel contains nonpositive realized variances.")
    return RealizedPanel(
        days, tuple(symbols), covariance, quarticity, graph_return, fwd_return,
        spy_cov[:, 0, 0], pd.Series({symbol: len(frame) for symbol, frame in frames.items()}),
    )
