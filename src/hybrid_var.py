"""
hybrid_var.py

Regime-switching Hybrid Market-and-Climate VaR — the central
methodological contribution of "Bridging Two Worlds" (Semar, 2025),
Appendix D.

Concept: rather than treating Market VaR (short horizon, statistical)
and Climate VaR (long horizon, scenario-based) as two separate,
incompatible frameworks, this model blends them into a single
short-horizon return distribution:

- With probability (1 - climate_prob), a simulated day/period follows
  ordinary market return dynamics.
- With probability climate_prob, a rescaled climate shock is added on
  top of the market return, representing the chance that a long-term
  transition/physical risk event materialises within the short
  holding period.

The climate shock itself is a long-horizon (e.g. to-2050) NGFS-scenario
shock, rescaled down to the VaR horizon in question (e.g. 5 days) under
a simplifying linear-attribution assumption — the dissertation's own
explicit simplification (see Appendix D), retained here rather than
silently "fixed", since the whole point is to reproduce the original
methodology faithfully.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from src.tail_risk import limited_expected_loss, weighted_limited_expected_loss


def rescale_climate_shock(
    long_horizon_shock: float,
    long_horizon_days: int = 25 * 252,  # e.g. ~25 years of trading days, to 2050
    target_horizon_days: int = 5,
) -> float:
    """
    Rescale a long-horizon climate shock (e.g. a 2050 NGFS scenario
    shock) down to a short target horizon (e.g. 5-day VaR), under a
    simplifying linear-attribution assumption:

        rescaled_shock = long_horizon_shock * (target_horizon_days / long_horizon_days)

    This is a simplification (flagged explicitly in the dissertation,
    Appendix D) — it assumes the climate shock accrues uniformly over
    time, when in reality transition risk is likely to be non-linear
    and lumpy. It's retained here as-is because reproducing the
    original, explicitly-caveated methodology is the point; a more
    sophisticated attribution curve is a natural extension.
    """
    return long_horizon_shock * (target_horizon_days / long_horizon_days)


def simulate_hybrid_returns(
    market_mean: float,
    market_std: float,
    climate_shock_rescaled: float,
    climate_prob: float = 0.15,
    n_simulations: int = 10_000,
    random_seed: int | None = 42,
) -> np.ndarray:
    """
    Simulate a regime-switching hybrid return distribution.

    Each simulated path draws a market return from N(market_mean,
    market_std), then with probability `climate_prob` adds the
    rescaled climate shock on top. climate_prob=0.15 matches the
    dissertation's own choice, representing "moderate stress" — treat
    it as a scenario assumption to sensitivity-test, not a fixed
    constant.
    """
    rng = np.random.default_rng(random_seed)

    market_returns = rng.normal(market_mean, market_std, n_simulations)
    regime_draws = rng.random(n_simulations)
    climate_regime = regime_draws <= climate_prob

    hybrid_returns = market_returns.copy()
    hybrid_returns[climate_regime] += climate_shock_rescaled

    return hybrid_returns


def hybrid_var_summary(
    market_returns: pd.Series,
    long_horizon_climate_shock: float,
    long_horizon_days: int = 25 * 252,
    target_horizon_days: int = 5,
    climate_prob: float = 0.15,
    confidence_levels: list[float] = [0.95, 0.99],
    n_simulations: int = 10_000,
) -> dict:
    """
    End-to-end Hybrid VaR pipeline for a single scenario: rescale the
    climate shock, simulate the regime-switching distribution, and
    extract VaR / LEL / WLEL — matching the dissertation's Tables 18/19.

    Parameters
    ----------
    market_returns : pd.Series
        Historical daily returns, used to parameterise the baseline
        market-regime mean and volatility.
    long_horizon_climate_shock : float
        A single NGFS-scenario-derived, beta-adjusted equity shock
        (e.g. from climate_var.climate_scenario_shock()), expressed
        over the long horizon (e.g. to 2050).
    """
    climate_shock_rescaled = rescale_climate_shock(
        long_horizon_climate_shock, long_horizon_days, target_horizon_days
    )

    # Scale market mean/std to the target horizon (e.g. 5-day) assuming
    # i.i.d. returns: mean scales linearly, std scales with sqrt(time).
    market_mean_scaled = market_returns.mean() * target_horizon_days
    market_std_scaled = market_returns.std() * np.sqrt(target_horizon_days)

    hybrid_returns = simulate_hybrid_returns(
        market_mean_scaled,
        market_std_scaled,
        climate_shock_rescaled,
        climate_prob,
        n_simulations,
    )

    results = {"climate_shock_rescaled_pct": climate_shock_rescaled * 100}
    for confidence in confidence_levels:
        alpha = 1 - confidence
        var_threshold = np.percentile(hybrid_returns, alpha * 100)
        lel = limited_expected_loss(hybrid_returns, var_threshold)
        wlel = weighted_limited_expected_loss(
            hybrid_returns, var_threshold, confidence, weighting="quadratic"
        )
        level_str = int(confidence * 100)
        results[f"hybrid_var_{level_str}_pct"] = -var_threshold * 100  # positive-loss convention for reporting
        results[f"lel_{level_str}_pct"] = -lel * 100
        results[f"wlel_{level_str}_pct"] = -wlel * 100

    return results


def run_hybrid_var_scenarios(
    market_returns: pd.Series,
    scenario_climate_shocks: dict[str, float],
    long_horizon_days: int = 25 * 252,
    target_horizon_days: int = 5,
    climate_prob: float = 0.15,
    confidence_levels: list[float] = [0.95, 0.99],
    n_simulations: int = 10_000,
) -> pd.DataFrame:
    """
    Run hybrid_var_summary() across multiple named scenarios (e.g.
    the output of climate_var.climate_scenario_shock() for each NGFS
    pathway) and return a summary table matching the dissertation's
    Tables 18/19.
    """
    rows = []
    for scenario_name, shock in scenario_climate_shocks.items():
        result = hybrid_var_summary(
            market_returns, shock, long_horizon_days, target_horizon_days,
            climate_prob, confidence_levels, n_simulations,
        )
        result["scenario"] = scenario_name
        rows.append(result)
    return pd.DataFrame(rows).set_index("scenario")
