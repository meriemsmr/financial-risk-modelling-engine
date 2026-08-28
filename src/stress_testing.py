"""
stress_testing.py

Scenario-based stress testing: apply defined shocks to portfolio assets
and calculate the resulting portfolio loss, independent of the
statistical VaR/ES framework. This is where "what does history say?"
gets replaced with "what if?".
"""

from __future__ import annotations
import pandas as pd


# Each scenario maps a ticker to a shock (simple % price move).
# Extend/edit these to build out your own scenario library.
SCENARIOS: dict[str, dict[str, float]] = {
    "covid_2020_crash": {
        "SPY": -0.20,
        "TLT": +0.08,    # flight to quality — bonds typically rally
        "GLD": +0.05,
        "USO": -0.65,    # oil was hit especially hard in the 2020 crash
    },
    "rate_shock_up_200bps": {
        "SPY": -0.08,
        "TLT": -0.15,    # long-duration bonds hit hardest by rate rises
        "GLD": -0.04,
        "USO": +0.02,
    },
    "ngfs_orderly_transition": {
        # Gradual, well-signalled climate policy tightening
        "SPY": -0.03,
        "TLT": +0.01,
        "GLD": +0.01,
        "USO": -0.10,
    },
    "ngfs_disorderly_transition": {
        # Late, abrupt climate policy action — larger, faster repricing
        "SPY": -0.12,
        "TLT": -0.03,
        "GLD": +0.06,
        "USO": -0.30,
    },
}


def apply_scenario(
    portfolio_weights: dict[str, float],
    scenario_name: str,
    scenarios: dict[str, dict[str, float]] = SCENARIOS,
) -> float:
    """
    Apply a named stress scenario to the portfolio and return the
    resulting portfolio-level loss (positive number = loss).

    Parameters
    ----------
    portfolio_weights : dict[str, float]
        e.g. {"SPY": 0.4, "TLT": 0.3, "GLD": 0.2, "USO": 0.1}
    scenario_name : str
        Must match a key in `scenarios`.
    """
    if scenario_name not in scenarios:
        raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(scenarios.keys())}")

    shocks = scenarios[scenario_name]
    portfolio_return = sum(
        portfolio_weights.get(ticker, 0) * shock
        for ticker, shock in shocks.items()
    )
    return -portfolio_return  # convert to positive-loss convention


def run_all_scenarios(
    portfolio_weights: dict[str, float],
    scenarios: dict[str, dict[str, float]] = SCENARIOS,
) -> pd.DataFrame:
    """
    Run every defined scenario against the portfolio and return a
    summary table sorted from worst to best — the table that belongs
    directly in the stress-testing section of the risk report.
    """
    results = [
        {"scenario": name, "portfolio_loss": apply_scenario(portfolio_weights, name, scenarios)}
        for name in scenarios
    ]
    return pd.DataFrame(results).sort_values("portfolio_loss", ascending=False).reset_index(drop=True)
