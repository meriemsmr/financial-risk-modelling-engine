"""
var.py

Value-at-Risk (Historical Simulation, Parametric, Monte Carlo) and
Expected Shortfall calculations.

Convention: VaR and Expected Shortfall are reported as positive numbers
representing a loss (e.g. VaR_95 = 0.023 means a 2.3% expected worst-case
loss at 95% confidence), consistent with standard risk-reporting practice.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm, t


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical Simulation VaR: the empirical loss at the given
    confidence percentile of the realised return distribution.

    "Based on previous market behaviour, what loss would we expect
    at the Xth percentile?"
    """
    alpha = 1 - confidence
    return -np.percentile(returns.dropna(), alpha * 100)


def parametric_var(
    returns: pd.Series,
    confidence: float = 0.95,
    dist: str = "normal",
    dof: int = 5,
) -> float:
    """
    Parametric VaR assuming a normal or Student-t return distribution.

    dist="t" is generally more appropriate for financial returns, which
    exhibit fatter tails than the normal distribution; `dof` sets the
    degrees of freedom for the t-distribution (lower = fatter tails).
    """
    mu = returns.mean()
    sigma = returns.std()
    alpha = 1 - confidence

    if dist == "normal":
        z = norm.ppf(alpha)
        return -(mu + z * sigma)
    elif dist == "t":
        t_quantile = t.ppf(alpha, dof)
        # Scale so the t-distribution has unit variance before applying sigma
        scaled_t = t_quantile * np.sqrt((dof - 2) / dof)
        return -(mu + scaled_t * sigma)
    else:
        raise ValueError("dist must be 'normal' or 't'")


def monte_carlo_var(
    returns: pd.Series,
    confidence: float = 0.95,
    n_simulations: int = 10_000,
    horizon: int = 1,
    random_seed: int | None = 42,
) -> float:
    """
    Monte Carlo VaR: simulate `n_simulations` possible future return
    paths (assuming normally distributed returns parameterised by the
    sample mean/std — swap in a fitted GARCH model's forecast for a
    more realistic version) and take the empirical percentile of
    simulated losses.
    """
    rng = np.random.default_rng(random_seed)
    mu = returns.mean()
    sigma = returns.std()

    simulated_returns = rng.normal(mu, sigma, size=(n_simulations, horizon)).sum(axis=1)
    alpha = 1 - confidence
    return -np.percentile(simulated_returns, alpha * 100)


def expected_shortfall(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """
    Expected Shortfall (Conditional VaR): the average loss in the tail
    beyond the VaR threshold. Answers "how bad are losses beyond the
    threshold?" rather than just "what is the threshold?"

    method="historical" uses the empirical tail average; extend with
    parametric/Monte Carlo variants following the same pattern as the
    VaR functions above.
    """
    alpha = 1 - confidence
    var_threshold = -historical_var(returns, confidence)  # back to raw quantile
    tail_losses = returns[returns <= var_threshold]
    return -tail_losses.mean()


def var_summary(returns: pd.Series, confidence_levels: list[float] = [0.95, 0.99]) -> pd.DataFrame:
    """
    Convenience function: produce a summary table of Historical,
    Parametric (normal + t) and Monte Carlo VaR, plus Expected
    Shortfall, across multiple confidence levels — the kind of table
    that belongs directly in the risk report.
    """
    rows = []
    for c in confidence_levels:
        rows.append({
            "confidence": c,
            "historical_var": historical_var(returns, c),
            "parametric_var_normal": parametric_var(returns, c, dist="normal"),
            "parametric_var_t": parametric_var(returns, c, dist="t"),
            "monte_carlo_var": monte_carlo_var(returns, c),
            "expected_shortfall": expected_shortfall(returns, c),
        })
    return pd.DataFrame(rows).set_index("confidence")


def var_summary_with_tail_risk(
    returns: pd.Series,
    confidence_levels: list[float] = [0.95, 0.99],
) -> pd.DataFrame:
    """
    Extended VaR summary that also reports LEL and WLEL for each
    method and confidence level, in the dissertation's own style
    (Tables 12-15) — i.e. LEL/WLEL computed on the *realised* return
    series relative to each method's VaR threshold, using the
    "confidence_level" WLEL weighting (WLEL = alpha * LEL).

    Note this mirrors the dissertation's Market VaR treatment of
    LEL/WLEL exactly; see src/tail_risk.py for the two weighting
    schemes and src/climate_var.py / src/hybrid_var.py for the
    "quadratic" weighting used on the climate side.
    """
    from src.tail_risk import limited_expected_loss, weighted_limited_expected_loss

    rows = []
    for c in confidence_levels:
        thresholds = {
            "historical": -historical_var(returns, c),
            "parametric_normal": -parametric_var(returns, c, dist="normal"),
            "parametric_t": -parametric_var(returns, c, dist="t"),
            "monte_carlo": -monte_carlo_var(returns, c),
        }
        for method, threshold in thresholds.items():
            lel = limited_expected_loss(returns, threshold)
            wlel = weighted_limited_expected_loss(returns, threshold, c, weighting="confidence_level")
            rows.append({
                "confidence": c, "method": method, "var_threshold": threshold,
                "LEL": lel, "WLEL": wlel,
            })
    return pd.DataFrame(rows).set_index(["confidence", "method"])
