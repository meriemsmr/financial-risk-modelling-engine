"""
earnings_var.py

Earnings-Event-Conditional VaR: the fusion point between my two
dissertations.

From the BSc dissertation ("Bridging Two Worlds"), this reuses the
regime-switching mechanic from hybrid_var.py — a scheduled event shifts
the simulated return distribution for that day, rather than treating
every day identically.

From the MSc/Citibank dissertation ("Reading Between the Lines"), this
reuses the evaluation discipline: don't just build the overlay, prove
whether it's actually justified. Specifically:

- Selectivity vs coverage framing, applied to *risk-model validation*
  rather than trading calls: does the score-adjusted VaR actually
  produce better-calibrated exceedances than plain VaR around earnings
  events, or does it just look more sophisticated?
- Threshold-free signal testing: correlate |score| against the
  magnitude of the realised earnings-day move (Spearman rank
  correlation), exactly the test that was the strongest evidence in
  the Citibank dissertation (rho = 0.2565, p = 0.0001).
- Honest reporting: if the overlay doesn't pass backtesting, that's the
  correct and reportable conclusion, not a bug to hide.

This is deliberately NOT positioned as an autonomous signal-generation
tool — consistent with both dissertations' own conclusions (BSc:
"illustrative rather than predictive"; MSc: "supported for triage,
not autonomous trading"). It's a risk *model validation* exercise:
does adding an earnings-aware overlay change (and specifically,
improve) how well a VaR model is calibrated around known event days?
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.tail_risk import limited_expected_loss, weighted_limited_expected_loss
from src.backtesting import compute_exceptions, kupiec_test, christoffersen_test


def event_conditional_var(
    baseline_mean: float,
    baseline_std: float,
    llm_score: float,
    score_to_return_scaling: float = 0.02,
    vol_inflation_factor: float = 1.5,
    confidence_levels: list[float] = [0.95, 0.99],
    n_simulations: int = 10_000,
    random_seed: int | None = 42,
) -> dict:
    """
    Monte Carlo VaR for a single earnings-event day, with the return
    distribution's mean shifted by the disclosure sentiment score and
    volatility inflated (earnings days empirically show elevated
    realised volatility versus ordinary trading days).

    Parameters
    ----------
    baseline_mean, baseline_std : float
        The asset's ordinary (non-event-day) return distribution
        parameters, e.g. from historical daily returns.
    llm_score : float
        Output of EarningsSignalScorer.score() (or a future real LLM
        call), roughly in [-1, 1].
    score_to_return_scaling : float
        Converts the sentiment score into an expected-return shift.
        Default 0.02 means a score of +1.0 shifts the expected return
        by +2 percentage points — a deliberately modest, tunable
        assumption; sensitivity-test this rather than treating it as
        fixed (same caveat-first spirit as the BSc dissertation's 15%
        climate-regime probability).
    vol_inflation_factor : float
        Multiplies baseline_std for the event day. 1.5 is a
        placeholder reflecting the well-documented fact that earnings
        days carry more uncertainty than average days; calibrate this
        empirically against your own event-day return data rather
        than trusting the default.
    """
    rng = np.random.default_rng(random_seed)

    event_mean = baseline_mean + (llm_score * score_to_return_scaling)
    event_std = baseline_std * vol_inflation_factor

    simulated_returns = rng.normal(event_mean, event_std, n_simulations)

    results = {"event_mean": event_mean, "event_std": event_std}
    for confidence in confidence_levels:
        alpha = 1 - confidence
        var_threshold = np.percentile(simulated_returns, alpha * 100)
        lel = limited_expected_loss(simulated_returns, var_threshold)
        wlel = weighted_limited_expected_loss(
            simulated_returns, var_threshold, confidence, weighting="quadratic"
        )
        level_str = int(confidence * 100)
        results[f"var_{level_str}"] = -var_threshold
        results[f"lel_{level_str}"] = -lel
        results[f"wlel_{level_str}"] = -wlel

    return results


def signal_return_correlation(
    scores: pd.Series,
    realised_returns: pd.Series,
) -> dict:
    """
    The single most important validation test, directly reproducing
    the Citibank dissertation's strongest piece of evidence (Section
    4.2): does the disclosure score's *direction and magnitude*
    actually track the realised market reaction, independent of any
    VaR threshold?

    Returns the Spearman rank correlation and p-value between the
    score series and the realised return series, aligned by event.
    """
    aligned = pd.concat([scores, realised_returns], axis=1, join="inner").dropna()
    aligned.columns = ["score", "return"]

    if len(aligned) < 3:
        return {"n": len(aligned), "rho": np.nan, "p_value": np.nan,
                "verdict": "Insufficient events to test"}

    rho, p_value = spearmanr(aligned["score"], aligned["return"])
    verdict = (
        f"Score tracks realised returns (rho={rho:.4f}, p={p_value:.4f})"
        if p_value < 0.05
        else f"No significant relationship detected (rho={rho:.4f}, p={p_value:.4f})"
    )
    return {"n": len(aligned), "rho": rho, "p_value": p_value, "verdict": verdict}


def backtest_event_overlay(
    event_returns: pd.Series,
    plain_var_series: pd.Series,
    overlay_var_series: pd.Series,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    The core validation exercise: does the sentiment-adjusted
    ("overlay") VaR produce better-calibrated exceedances on earnings-
    event days than plain statistical VaR, using the same Kupiec and
    Christoffersen tests from backtesting.py?

    This directly answers the dissertation-consistent question: is the
    added complexity of the overlay actually justified by the data, or
    does it just look more sophisticated? Report both results side by
    side — a null result here ("no improvement") is a legitimate and
    reportable finding, not a failure of the exercise.

    Parameters
    ----------
    event_returns : pd.Series
        Realised returns on earnings-event days only.
    plain_var_series, overlay_var_series : pd.Series
        VaR estimates (positive-loss convention) for the same event
        days, from the baseline model and the score-adjusted model
        respectively.
    """
    rows = []
    for label, var_series in [("plain_var", plain_var_series), ("overlay_var", overlay_var_series)]:
        exceptions = compute_exceptions(event_returns, var_series).dropna()
        kupiec = kupiec_test(exceptions, confidence)
        christoffersen = christoffersen_test(exceptions) if len(exceptions) > 1 else {
            "lr_statistic": np.nan, "p_value": np.nan, "verdict": "Insufficient data"
        }
        rows.append({
            "model": label,
            "n_events": kupiec["n_observations"],
            "n_exceptions": kupiec["n_exceptions"],
            "observed_rate": kupiec["observed_rate"],
            "kupiec_p_value": kupiec["p_value"],
            "kupiec_verdict": kupiec["verdict"],
            "christoffersen_p_value": christoffersen["p_value"],
            "christoffersen_verdict": christoffersen["verdict"],
        })
    return pd.DataFrame(rows).set_index("model")
