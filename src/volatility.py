"""
volatility.py

Volatility forecasting models: Historical, EWMA, GARCH(1,1) and EGARCH.
Each function returns a fitted volatility series (and, where relevant,
the fitted model object for forecasting out-of-sample).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from arch import arch_model


def historical_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    """
    Rolling-window realised (historical) volatility — the naive
    benchmark every other model should beat.

    Annualisation note: multiply by sqrt(252) if you want annualised
    vol; left as daily here for direct comparability with the other
    models below.
    """
    return returns.rolling(window=window).std()


def ewma_volatility(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """
    Exponentially Weighted Moving Average volatility (RiskMetrics-style):

        sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_{t-1}^2

    lam=0.94 is the standard RiskMetrics daily decay factor.
    """
    squared_returns = returns**2
    ewma_var = squared_returns.ewm(alpha=(1 - lam), adjust=False).mean()
    return np.sqrt(ewma_var)


def fit_garch(returns: pd.Series, p: int = 1, q: int = 1):
    """
    Fit a GARCH(p, q) model (default GARCH(1,1)) to a return series
    using the `arch` package. Returns the fitted model result object,
    from which conditional volatility and out-of-sample forecasts can
    be extracted.

    Note: `arch` expects returns scaled to roughly percentage points
    (e.g. returns * 100) for numerical stability — handle that before
    calling this function, or scale internally and document clearly.
    """
    scaled_returns = returns * 100
    model = arch_model(scaled_returns, vol="GARCH", p=p, q=q, dist="normal")
    fitted = model.fit(disp="off")
    return fitted


def fit_egarch(returns: pd.Series, p: int = 1, o: int = 1, q: int = 1):
    """
    Fit an EGARCH(p, o, q) model, which captures the asymmetric
    ("leverage") effect: volatility tends to rise more after negative
    returns than after positive returns of the same magnitude.
    """
    scaled_returns = returns * 100
    model = arch_model(scaled_returns, vol="EGARCH", p=p, o=o, q=q, dist="normal")
    fitted = model.fit(disp="off")
    return fitted


def forecast_volatility(fitted_model, horizon: int = 1) -> pd.Series:
    """
    Produce an out-of-sample volatility forecast from a fitted GARCH or
    EGARCH model (as returned by fit_garch / fit_egarch).
    """
    forecast = fitted_model.forecast(horizon=horizon, reindex=False)
    return np.sqrt(forecast.variance.iloc[-1]) / 100  # unscale back from percentage


def evaluate_forecasts(actual: pd.Series, predicted: pd.Series) -> dict:
    """
    Compare forecast accuracy across volatility models using RMSE, MAE
    and QLIKE loss (the standard loss function for volatility forecast
    evaluation, penalising under-prediction of volatility more heavily
    than over-prediction).
    """
    aligned = pd.concat([actual, predicted], axis=1, join="inner").dropna()
    a, p = aligned.iloc[:, 0], aligned.iloc[:, 1]

    rmse = np.sqrt(np.mean((a - p) ** 2))
    mae = np.mean(np.abs(a - p))
    qlike = np.mean(np.log(p**2) + (a**2) / (p**2))

    return {"RMSE": rmse, "MAE": mae, "QLIKE": qlike}
