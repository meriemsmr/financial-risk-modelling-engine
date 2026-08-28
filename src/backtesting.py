"""
backtesting.py

VaR model validation: does the model's stated confidence level actually
hold up against realised outcomes? Implements the two standard
regulatory backtests:

- Kupiec Proportion of Failures (POF) test: is the exception rate correct?
- Christoffersen independence test: are exceptions randomly distributed,
  or clustered in time (a sign the model fails to adapt during
  volatile regimes)?
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import chi2


def compute_exceptions(returns: pd.Series, var_series: pd.Series) -> pd.Series:
    """
    Compare realised losses against a (possibly time-varying) VaR
    series and flag "exceptions": days where the realised loss
    exceeded the VaR estimate.

    Both series should be aligned by date and expressed as positive-
    loss convention (see var.py docstring).
    """
    losses = -returns  # convert returns to losses
    aligned = pd.concat([losses, var_series], axis=1, join="inner").dropna()
    aligned.columns = ["loss", "var"]
    return (aligned["loss"] > aligned["var"]).astype(int)


def kupiec_test(exceptions: pd.Series, confidence: float = 0.95) -> dict:
    """
    Kupiec (1995) Proportion of Failures test.

    H0: the observed exception rate equals the expected exception rate
    (1 - confidence). Uses a likelihood-ratio statistic, asymptotically
    chi-squared with 1 degree of freedom.

    Returns the LR statistic, p-value, and a plain-English verdict.
    """
    n = len(exceptions)
    x = exceptions.sum()  # number of exceptions observed
    p_expected = 1 - confidence
    p_observed = x / n if n > 0 else np.nan

    if x == 0 or x == n:
        # Avoid log(0) at the boundary
        lr_stat = np.nan
        p_value = np.nan
    else:
        lr_stat = -2 * (
            (n - x) * np.log(1 - p_expected) + x * np.log(p_expected)
            - (n - x) * np.log(1 - p_observed) - x * np.log(p_observed)
        )
        p_value = 1 - chi2.cdf(lr_stat, df=1)

    verdict = (
        "Fail to reject H0 — exception rate consistent with model"
        if (p_value is not np.nan and p_value > 0.05)
        else "Reject H0 — exception rate inconsistent with model"
    )

    return {
        "n_observations": n,
        "n_exceptions": int(x),
        "expected_rate": p_expected,
        "observed_rate": p_observed,
        "lr_statistic": lr_stat,
        "p_value": p_value,
        "verdict": verdict,
    }


def christoffersen_test(exceptions: pd.Series) -> dict:
    """
    Christoffersen (1998) test of independence: tests whether
    exceptions are independently distributed through time, or cluster
    together (e.g. several breaches in a row during a volatile
    stretch — a sign the model isn't adapting quickly enough).

    Builds a transition matrix (exception at t-1 -> exception at t)
    and tests independence via a likelihood-ratio statistic.
    """
    exc = exceptions.values
    n00 = n01 = n10 = n11 = 0

    for i in range(1, len(exc)):
        prev, curr = exc[i - 1], exc[i]
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        elif prev == 1 and curr == 1:
            n11 += 1

    n0, n1 = n00 + n01, n10 + n11
    if n0 == 0 or n1 == 0 or (n01 + n11) == 0:
        return {"lr_statistic": np.nan, "p_value": np.nan, "verdict": "Insufficient exceptions to test"}

    pi0 = n01 / n0 if n0 > 0 else 0
    pi1 = n11 / n1 if n1 > 0 else 0
    pi = (n01 + n11) / (n0 + n1)

    def _log_lik(p, successes, trials):
        if p in (0, 1) or trials == 0:
            return 0
        return successes * np.log(p) + (trials - successes) * np.log(1 - p)

    ll_unrestricted = _log_lik(pi0, n01, n0) + _log_lik(pi1, n11, n1)
    ll_restricted = _log_lik(pi, n01 + n11, n0 + n1)

    lr_stat = -2 * (ll_restricted - ll_unrestricted)
    p_value = 1 - chi2.cdf(lr_stat, df=1)

    verdict = (
        "Fail to reject H0 — exceptions appear independent"
        if p_value > 0.05
        else "Reject H0 — exceptions are clustered (model may lag volatile regimes)"
    )

    return {"lr_statistic": lr_stat, "p_value": p_value, "verdict": verdict}
