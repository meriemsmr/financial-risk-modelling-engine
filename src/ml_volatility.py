"""
ml_volatility.py

Part 8 (optional extension) of the original project plan: does a
machine-learning model actually forecast volatility better than the
classical GARCH-family models in volatility.py — or does the added
complexity not earn its keep?

This is deliberately built LAST and treated as a genuine research
question with an uncertain answer, not a foregone "ML wins" exercise.
Consistent with the honesty standard carried through the rest of this
project (see earnings_var.py, hybrid_var.py docstrings): if GARCH wins,
that is the correct and reportable finding.

Features used to predict next-day realised volatility:
- lagged returns (autocorrelation / momentum effects)
- lagged realised volatility (volatility clustering — the same stylised
  fact GARCH is built to capture)
- rolling skewness / kurtosis (distributional shape signals)
- a volume proxy, if available (activity often leads volatility)

Models compared:
- GARCH(1,1) forecast (from volatility.py) — the classical baseline
- Random Forest regressor
- XGBoost regressor

Evaluated on the same out-of-sample metrics as volatility.py
(RMSE, MAE, QLIKE), so results are directly comparable across the
whole project rather than using a different yardstick for the ML piece.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from src.volatility import evaluate_forecasts


def build_features(
    returns: pd.Series,
    realised_vol_window: int = 5,
    n_lags: int = 5,
) -> pd.DataFrame:
    """
    Build a feature matrix for ML volatility forecasting.

    Target (added separately in build_target()) is next-day realised
    volatility; features here are all constructed to be available at
    the time of prediction (no look-ahead).

    Parameters
    ----------
    returns : pd.Series
        Daily log returns.
    realised_vol_window : int
        Rolling window for the realised-volatility feature and target.
    n_lags : int
        Number of lagged return observations to include as features.
    """
    df = pd.DataFrame(index=returns.index)

    # Lagged returns (momentum / autocorrelation signal)
    for lag in range(1, n_lags + 1):
        df[f"return_lag_{lag}"] = returns.shift(lag)

    # Lagged realised volatility (the volatility-clustering feature —
    # same stylised fact GARCH exploits, given directly to the ML model)
    realised_vol = returns.rolling(realised_vol_window).std()
    for lag in range(1, n_lags + 1):
        df[f"realised_vol_lag_{lag}"] = realised_vol.shift(lag)

    # Rolling distributional shape features
    df["rolling_skew"] = returns.rolling(21).skew().shift(1)
    df["rolling_kurtosis"] = returns.rolling(21).kurt().shift(1)

    # Absolute lagged return (captures magnitude regardless of direction —
    # complements signed lagged returns above)
    df["abs_return_lag_1"] = returns.shift(1).abs()

    return df


def build_target(returns: pd.Series, realised_vol_window: int = 5) -> pd.Series:
    """
    Target variable: forward-looking realised volatility over the next
    `realised_vol_window` days, computed as of each date (i.e. shifted
    so it represents genuinely future information relative to the
    features in build_features()).
    """
    realised_vol = returns.rolling(realised_vol_window).std()
    return realised_vol.shift(-realised_vol_window)


def train_test_split_time_series(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2
) -> tuple:
    """
    Chronological (non-shuffled) train/test split — critical for time
    series data, where a random shuffle would leak future information
    into the training set.
    """
    aligned = pd.concat([X, y.rename("target")], axis=1).dropna()
    split_idx = int(len(aligned) * (1 - test_size))

    train, test = aligned.iloc[:split_idx], aligned.iloc[split_idx:]
    X_train, y_train = train.drop(columns="target"), train["target"]
    X_test, y_test = test.drop(columns="target"), test["target"]
    return X_train, X_test, y_train, y_test


def fit_random_forest(X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> RandomForestRegressor:
    """Fit a Random Forest regressor with sensible defaults for a small
    financial dataset (shallow-ish trees to reduce overfitting risk)."""
    params = dict(n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=42)
    params.update(kwargs)
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model


def fit_xgboost(X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> xgb.XGBRegressor:
    """Fit an XGBoost regressor with conservative defaults (shallow
    depth, moderate learning rate) to reduce overfitting on a
    relatively small, noisy financial dataset."""
    params = dict(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
    params.update(kwargs)
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model


def compare_ml_vs_garch(
    returns: pd.Series,
    garch_forecast: pd.Series,
    realised_vol_window: int = 5,
    n_lags: int = 5,
    test_size: float = 0.2,
) -> pd.DataFrame:
    """
    End-to-end comparison: build features/target, fit Random Forest and
    XGBoost, and evaluate all three approaches (including the supplied
    GARCH forecast) on the same out-of-sample test period using
    evaluate_forecasts() from volatility.py — so every model in this
    project is judged by the same yardstick.

    Parameters
    ----------
    returns : pd.Series
        Daily log returns.
    garch_forecast : pd.Series
        A GARCH conditional-volatility series aligned to `returns`
        (e.g. from fit_garch() in volatility.py), used as the
        classical baseline.
    """
    X = build_features(returns, realised_vol_window, n_lags)
    y = build_target(returns, realised_vol_window)
    X_train, X_test, y_train, y_test = train_test_split_time_series(X, y, test_size)

    rf_model = fit_random_forest(X_train, y_train)
    xgb_model = fit_xgboost(X_train, y_train)

    rf_pred = pd.Series(rf_model.predict(X_test), index=X_test.index)
    xgb_pred = pd.Series(xgb_model.predict(X_test), index=X_test.index)

    # Align GARCH forecast to the same test period for a fair comparison
    garch_test = garch_forecast.reindex(y_test.index)

    results = {
        "GARCH(1,1)": evaluate_forecasts(y_test, garch_test),
        "Random Forest": evaluate_forecasts(y_test, rf_pred),
        "XGBoost": evaluate_forecasts(y_test, xgb_pred),
    }
    return pd.DataFrame(results).T


def feature_importance_report(model, feature_names: list[str]) -> pd.Series:
    """
    Return feature importances from a fitted RF/XGBoost model, sorted
    descending — useful for checking whether the model is actually
    learning something sensible (e.g. lagged realised volatility should
    typically dominate) versus overfitting to noise.
    """
    importances = pd.Series(model.feature_importances_, index=feature_names)
    return importances.sort_values(ascending=False)
