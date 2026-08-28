"""
data_loader.py

Downloads, cleans, and processes market price data for the risk-modelling
engine. Converts raw adjusted-close prices into log returns, the base
input for every downstream volatility, VaR and stress-testing module.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import yfinance as yf


def download_prices(
    tickers: list[str],
    start: str = "2015-01-01",
    end: str | None = None,
    cache_path: str = "data/market_prices.csv",
) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers via yfinance,
    cache to CSV, and return a wide DataFrame (date index, one column
    per ticker).

    Parameters
    ----------
    tickers : list[str]
        e.g. ["SPY", "TLT", "GLD", "USO"]
    start, end : str
        Date range in "YYYY-MM-DD" format. `end=None` defaults to today.
    cache_path : str
        Where to save/read the cached CSV, so repeat runs don't re-hit
        the API.

    Returns
    -------
    pd.DataFrame
        Wide-format adjusted close prices, indexed by date.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    raw = raw.sort_index()
    raw.to_csv(cache_path)
    return raw


def load_cached_prices(cache_path: str = "data/market_prices.csv") -> pd.DataFrame:
    """Load previously cached price data instead of re-downloading."""
    return pd.read_csv(cache_path, index_col=0, parse_dates=True)


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a wide price DataFrame into log returns:
        r_t = ln(P_t / P_{t-1})

    Drops the first row (undefined return) and any rows with all-NaN
    values (e.g. holidays where a subset of tickers didn't trade).
    """
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.dropna(how="all").iloc[1:]


def load_portfolio_returns(
    tickers: list[str],
    start: str = "2015-01-01",
    end: str | None = None,
    use_cache: bool = False,
) -> pd.DataFrame:
    """
    Convenience wrapper: download (or load cached) prices and return
    log returns in one call.
    """
    if use_cache:
        prices = load_cached_prices()
    else:
        prices = download_prices(tickers, start=start, end=end)
    return compute_log_returns(prices)


def portfolio_returns(
    asset_returns: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """
    Combine individual asset log returns into a single portfolio return
    series using specified weights (equal-weighted by default).

    Note: for log returns this is an approximation (true portfolio
    returns should be computed from simple returns and converted back);
    fine for exploratory analysis, but revisit before using in the VaR
    engine if precision matters.
    """
    if weights is None:
        weights = {col: 1 / len(asset_returns.columns) for col in asset_returns.columns}
    w = pd.Series(weights)
    return (asset_returns[w.index] * w).sum(axis=1)
