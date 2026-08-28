"""
climate_var.py

Climate Value-at-Risk: translates long-horizon NGFS transition
scenario shocks into firm-specific, beta-adjusted equity return
distributions via Monte Carlo simulation, exactly following the
methodology in "Bridging Two Worlds" (Semar, 2025), Appendix C.

Steps:
1. Estimate firm-specific beta via CAPM (equity excess returns
   regressed on market excess returns).
2. Apply each NGFS scenario's % market shock, scaled by beta, to get
   a scenario-adjusted expected equity return.
3. Use that adjusted mean (with the firm's own historical volatility)
   as the input to a Monte Carlo simulation of equity returns under
   each scenario.
4. Extract VaR, LEL and WLEL from each scenario's simulated
   distribution.

Note on NGFS data: NGFS_SCENARIOS below is real data downloaded from the
NGFS Scenario Explorer (https://www.ngfs.net/ngfs-scenarios-portal/),
specifically the GCAM 6.0 NGFS model's World GDP|MER (Counterfactual
without damage) series — used as a GDP-based proxy for market-level
impact, following Dietz et al. (2016)'s approach (cited in the BSc
dissertation), since NGFS does not publish equity-index projections
directly. Re-download and update if the NGFS portal releases a newer
scenario vintage (this snapshot is Phase 5).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.tail_risk import limited_expected_loss, weighted_limited_expected_loss


# Real NGFS Phase 5 GDP data, downloaded from the NGFS Scenario Explorer
# (https://data.ene.iiasa.ac.at/ngfs), Model: GCAM 6.0 NGFS, Region: World,
# Variable: GDP|MER|Counterfactual without damage (i.e. transition risk only,
# excluding physical/chronic damage). Values are the ratio of 2050 GDP to
# 2025 GDP under each scenario (e.g. 1.6763 = GDP grows to 167.63% of its
# 2025 level by 2050 under that pathway) — used as the market-level "shock"
# input to climate_scenario_shock(), following the same GDP-as-market-proxy
# approach as Dietz et al. (2016), cited in the BSc dissertation.
NGFS_SCENARIOS: dict[str, float] = {
    "Baseline": 1.7247,              # NGFS "Current Policies" scenario
    "Below_2C": 1.6763,
    "Delayed_Transition": 1.6481,
    "Fragmented_World": 1.7042,
    "NDCs": 1.7100,
    "Net_Zero_2050": 1.6101,
}

# Kept for backward compatibility with earlier notebooks/tests that
# reference the old placeholder name — now points to the same real data.
SAMPLE_NGFS_SCENARIOS = NGFS_SCENARIOS


def estimate_capm_beta(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: pd.Series | float = 0.0,
) -> float:
    """
    Estimate an asset's CAPM beta via OLS regression of excess asset
    returns on excess market returns:

        (r_asset - r_f) = alpha + beta * (r_market - r_f) + epsilon

    risk_free_rate can be a constant (e.g. 0.0 to ignore) or a Series
    aligned with the return data (e.g. a daily T-bill yield).
    """
    aligned = pd.concat([asset_returns, market_returns], axis=1, join="inner").dropna()
    aligned.columns = ["asset", "market"]

    if isinstance(risk_free_rate, pd.Series):
        aligned = aligned.join(risk_free_rate.rename("rf"), how="inner").dropna()
        excess_asset = aligned["asset"] - aligned["rf"]
        excess_market = aligned["market"] - aligned["rf"]
    else:
        excess_asset = aligned["asset"] - risk_free_rate
        excess_market = aligned["market"] - risk_free_rate

    X = sm.add_constant(excess_market)
    model = sm.OLS(excess_asset, X).fit()
    return model.params.iloc[1]  # beta coefficient


def climate_scenario_shock(scenario_market_change: float, beta: float) -> float:
    """
    Translate a scenario's aggregate market % change into a
    firm-specific expected return shock, by applying the firm's beta:

        firm_shock = beta * market_change

    Note: scenario_market_change should already be expressed relative
    to baseline (e.g. 0.85 means the market index is projected at 85%
    of where it otherwise would have been) — subtract 1.0 first if
    your NGFS data is in raw index-level terms rather than % change.
    """
    return beta * scenario_market_change


def monte_carlo_climate_var(
    expected_return: float,
    historical_volatility: float,
    confidence_levels: list[float] = [0.95, 0.99],
    n_simulations: int = 10_000,
    random_seed: int | None = 42,
) -> dict:
    """
    Run a Monte Carlo simulation of equity returns under a single
    climate scenario: draws n_simulations returns from a normal
    distribution centred on the scenario's beta-adjusted expected
    return, using the firm's own historical volatility (the
    dissertation's approach — climate scenarios shift the mean, not
    the variance, reflecting structural/policy risk rather than
    short-term volatility dynamics).

    Returns VaR, LEL and WLEL (quadratic weighting, matching Appendix C)
    at each requested confidence level.
    """
    rng = np.random.default_rng(random_seed)
    simulated_returns = rng.normal(expected_return, historical_volatility, n_simulations)

    results = {}
    for confidence in confidence_levels:
        alpha = 1 - confidence
        var_threshold = np.percentile(simulated_returns, alpha * 100)
        lel = limited_expected_loss(simulated_returns, var_threshold)
        wlel = weighted_limited_expected_loss(
            simulated_returns, var_threshold, confidence, weighting="quadratic"
        )
        results[confidence] = {
            "var": var_threshold,
            "lel": lel,
            "wlel": wlel,
        }
    return results


def run_climate_var_scenarios(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    scenarios: dict[str, float] = SAMPLE_NGFS_SCENARIOS,
    risk_free_rate: pd.Series | float = 0.0,
    confidence_levels: list[float] = [0.95, 0.99],
    n_simulations: int = 10_000,
) -> pd.DataFrame:
    """
    End-to-end Climate VaR pipeline: estimate beta, apply each
    scenario's shock, run Monte Carlo, and return a summary table in
    the same shape as the dissertation's Tables 16/17.
    """
    beta = estimate_capm_beta(asset_returns, market_returns, risk_free_rate)
    historical_vol = asset_returns.std()

    rows = []
    for scenario_name, market_change in scenarios.items():
        expected_return = climate_scenario_shock(market_change, beta)
        mc_results = monte_carlo_climate_var(
            expected_return, historical_vol, confidence_levels, n_simulations
        )

        row = {"scenario": scenario_name, "beta": beta, "expected_return_pct": expected_return * 100}
        for confidence, metrics in mc_results.items():
            level_str = int(confidence * 100)
            row[f"var_{level_str}_pct"] = metrics["var"] * 100
            row[f"lel_{level_str}_pct"] = metrics["lel"] * 100
            row[f"wlel_{level_str}_pct"] = metrics["wlel"] * 100
        rows.append(row)

    return pd.DataFrame(rows).set_index("scenario")
