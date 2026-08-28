"""
tail_risk.py

Limited Expected Loss (LEL) and Weighted Limited Expected Loss (WLEL) —
the tail-risk metrics extended in "Bridging Two Worlds: A Comparative
Analysis of Market VaR and Climate VaR Enhanced by Tail-Loss Metrics
(LEL/WLEL)" (Semar, 2025).

VaR answers "what loss should I expect at this confidence level?"
LEL answers "if I'm in the tail beyond that threshold, how bad is it
on average?"
WLEL goes further, up-weighting the most extreme losses within the
tail so that rare, catastrophic outcomes aren't averaged away — this
is the dissertation's own methodological extension of Basak & Shapiro
(2001)'s LEL concept, following the weighting logic in Chen & Nguyen
(2024).

These are written generically so the same functions serve both the
Market VaR module (var.py) and the Climate VaR module
(climate_var.py) — exactly as in the original dissertation, where LEL
and WLEL were computed identically for both domains.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def limited_expected_loss(
    returns: pd.Series | np.ndarray,
    var_threshold: float,
) -> float:
    """
    Limited Expected Loss (LEL): the average realised/simulated return
    conditional on being worse than the VaR threshold.

    Parameters
    ----------
    returns : Series or array
        Realised returns (Market VaR use case) or simulated returns
        (Climate VaR / Monte Carlo use case). Raw returns, NOT the
        positive-loss convention used elsewhere in this project —
        this module follows the dissertation's own sign convention
        (negative = loss) to stay faithful to the original methodology.
    var_threshold : float
        The VaR cutoff, in the same raw-return sign convention (e.g.
        -0.02 for a 2% loss threshold).

    Returns
    -------
    float
        The average return in the tail beyond var_threshold. Negative
        = an expected loss, consistent with the dissertation's tables.
    """
    returns = pd.Series(returns).dropna()
    tail = returns[returns < var_threshold]
    if len(tail) == 0:
        return np.nan
    return tail.mean()


def weighted_limited_expected_loss(
    returns: pd.Series | np.ndarray,
    var_threshold: float,
    confidence: float = 0.99,
    weighting: str = "confidence_level",
) -> float:
    """
    Weighted Limited Expected Loss (WLEL).

    Two weighting schemes are supported:

    - "confidence_level" (default): reproduces the dissertation's own
      EViews implementation, where WLEL = (1 - confidence) * LEL — a
      simple scaling by the tail probability itself (e.g. WLEL_99% =
      0.01 * LEL_99%). This is the exact formula used throughout the
      dissertation's Market VaR tables (Appendix E).

    - "quadratic": the alternative weighting scheme described in
      Appendix D for the Climate VaR / Hybrid VaR tail metrics, which
      applies squared-loss weights within the tail so the most extreme
      simulated outcomes dominate the average more than a simple mean
      (i.e. WLEL = sum(w_i * r_i) / sum(w_i), where w_i = |r_i|^2 for
      each simulated return r_i in the tail).

    Parameters
    ----------
    returns, var_threshold : see limited_expected_loss()
    confidence : float
        Required for the "confidence_level" scheme (e.g. 0.99 for a
        99% VaR threshold).
    weighting : {"confidence_level", "quadratic"}
        Which of the dissertation's two WLEL formulas to use. Use
        "confidence_level" for Market VaR (matches Appendix E exactly);
        use "quadratic" for Climate VaR / Hybrid VaR (matches
        Appendices C and D).
    """
    returns = pd.Series(returns).dropna()
    tail = returns[returns < var_threshold]

    if len(tail) == 0:
        return np.nan

    if weighting == "confidence_level":
        lel = tail.mean()
        alpha = 1 - confidence
        return alpha * lel

    elif weighting == "quadratic":
        weights = tail.abs() ** 2
        if weights.sum() == 0:
            return np.nan
        return (weights * tail).sum() / weights.sum()

    else:
        raise ValueError("weighting must be 'confidence_level' or 'quadratic'")


def tail_risk_summary(
    returns: pd.Series,
    var_thresholds: dict[str, float],
    weighting: str = "confidence_level",
) -> pd.DataFrame:
    """
    Convenience function: compute LEL and WLEL across multiple VaR
    thresholds/methods in one call, producing a table in the same
    shape as the dissertation's Tables 12–15.

    Parameters
    ----------
    returns : pd.Series
        Return series (raw sign convention — negative = loss).
    var_thresholds : dict[str, float]
        e.g. {"parametric_95": -0.0165, "historical_95": -0.019, ...}
        Keys should encode both method and confidence level so the
        correct alpha can be inferred for "confidence_level" weighting.
    weighting : str
        See weighted_limited_expected_loss().
    """
    rows = []
    for label, threshold in var_thresholds.items():
        # infer confidence from the label suffix, defaulting to 0.95
        confidence = 0.99 if "99" in label else 0.95
        lel = limited_expected_loss(returns, threshold)
        wlel = weighted_limited_expected_loss(returns, threshold, confidence, weighting)
        rows.append({"method": label, "var_threshold": threshold, "LEL": lel, "WLEL": wlel})
    return pd.DataFrame(rows).set_index("method")
