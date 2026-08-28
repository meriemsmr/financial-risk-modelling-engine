"""
app.py

Streamlit web interface for the Financial Risk Modelling Engine.

Run locally with:
    streamlit run app.py

This is a thin UI layer over the tested src/ modules — every
calculation here calls straight into volatility.py, var.py,
backtesting.py, stress_testing.py, climate_var.py, hybrid_var.py,
earnings_signal.py, earnings_var.py and ml_volatility.py. No modelling
logic lives in this file; it only wires those modules to widgets and
charts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(__file__))

from src.data_loader import download_prices, compute_log_returns, portfolio_returns
from src.volatility import (
    historical_volatility, ewma_volatility, fit_garch, fit_egarch,
    forecast_volatility, evaluate_forecasts,
)
from src.var import var_summary, var_summary_with_tail_risk
from src.backtesting import compute_exceptions, kupiec_test, christoffersen_test
from src.stress_testing import run_all_scenarios, SCENARIOS
from src.climate_var import estimate_capm_beta, run_climate_var_scenarios, climate_scenario_shock, SAMPLE_NGFS_SCENARIOS
from src.hybrid_var import run_hybrid_var_scenarios
from src.earnings_signal import EarningsSignalScorer
from src.earnings_var import event_conditional_var, signal_return_correlation
from src.ml_volatility import compare_ml_vs_garch, build_features, build_target, train_test_split_time_series, fit_random_forest, feature_importance_report


st.set_page_config(page_title="Financial Risk Modelling Engine", layout="wide")

DEFAULT_TICKERS = ["SPY", "TLT", "GLD", "USO"]

st.title("📊 Financial Risk Modelling Engine")
st.caption(
    "Volatility forecasting · VaR · Expected Shortfall · Backtesting · Stress testing · "
    "Climate VaR · Hybrid regime-switching VaR · Earnings-event NLP overlay · ML vs GARCH"
)

# ---------------------------------------------------------------------------
# Sidebar: data loading (shared across all tabs via session_state)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Portfolio")
    tickers = st.multiselect("Tickers", DEFAULT_TICKERS, default=DEFAULT_TICKERS)
    start_date = st.date_input("Start date", pd.Timestamp("2015-01-01"))
    weights_equal = st.checkbox("Equal-weight portfolio", value=True)

    if st.button("Load data", type="primary"):
        with st.spinner("Downloading price data..."):
            try:
                prices = download_prices(tickers, start=str(start_date), cache_path="data/market_prices.csv")
                returns = compute_log_returns(prices)
                weights = {t: 1 / len(tickers) for t in tickers} if weights_equal else None
                port_ret = portfolio_returns(returns, weights) if weights else returns.iloc[:, 0]

                st.session_state["prices"] = prices
                st.session_state["returns"] = returns
                st.session_state["port_ret"] = port_ret
                st.session_state["weights"] = weights or {tickers[0]: 1.0}
                st.success(f"Loaded {len(prices)} days of data.")
            except Exception as e:
                st.error(f"Data load failed: {e}")

    if "port_ret" in st.session_state:
        st.info(f"{len(st.session_state['port_ret'])} return observations loaded.")
    else:
        st.warning("Click 'Load data' to begin — every tab below needs this first.")

    st.divider()
    st.caption("Fusion of my BSc VaR dissertation and MSc Citibank LLM dissertation. "
               "See the README for full methodology and citations.")


def require_data():
    if "port_ret" not in st.session_state:
        st.warning("Load portfolio data from the sidebar first.")
        st.stop()


tabs = st.tabs([
    "1. Data & Returns", "2. Volatility", "3. VaR & ES", "4. Backtesting",
    "5. Stress Testing", "6. Climate VaR", "7. Hybrid VaR",
    "8. Earnings Overlay", "9. ML vs GARCH",
])

# ---------------------------------------------------------------------------
# TAB 1 — Data & Returns
# ---------------------------------------------------------------------------
with tabs[0]:
    require_data()
    returns = st.session_state["returns"]
    port_ret = st.session_state["port_ret"]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cumulative Returns")
        fig, ax = plt.subplots(figsize=(7, 4))
        (1 + returns).cumprod().plot(ax=ax)
        ax.set_ylabel("Growth of $1")
        st.pyplot(fig)

    with col2:
        st.subheader("Correlation Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(returns.corr(), cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(returns.columns))); ax.set_xticklabels(returns.columns)
        ax.set_yticks(range(len(returns.columns))); ax.set_yticklabels(returns.columns)
        for i in range(len(returns.columns)):
            for j in range(len(returns.columns)):
                ax.text(j, i, f"{returns.corr().iloc[i, j]:.2f}", ha="center", va="center")
        fig.colorbar(im)
        st.pyplot(fig)

    st.subheader("Descriptive Statistics")
    st.dataframe(returns.describe().round(5))

# ---------------------------------------------------------------------------
# TAB 2 — Volatility Models
# ---------------------------------------------------------------------------
with tabs[1]:
    require_data()
    port_ret = st.session_state["port_ret"]

    st.subheader("Historical vs EWMA vs GARCH vs EGARCH")
    with st.spinner("Fitting volatility models..."):
        hist_vol = historical_volatility(port_ret)
        ewma_vol = ewma_volatility(port_ret)
        garch_fit = fit_garch(port_ret)
        egarch_fit = fit_egarch(port_ret)

        garch_vol = pd.Series(garch_fit.conditional_volatility / 100, index=port_ret.index)
        egarch_vol = pd.Series(egarch_fit.conditional_volatility / 100, index=port_ret.index)

    vol_df = pd.DataFrame({
        "Historical": hist_vol, "EWMA": ewma_vol, "GARCH": garch_vol, "EGARCH": egarch_vol,
    })
    fig, ax = plt.subplots(figsize=(11, 4))
    vol_df.plot(ax=ax)
    ax.set_ylabel("Daily volatility")
    st.pyplot(fig)

    st.caption("GARCH and EGARCH model summaries available on request — see notebooks/02_volatility_models.ipynb for full diagnostics.")

# ---------------------------------------------------------------------------
# TAB 3 — VaR & Expected Shortfall
# ---------------------------------------------------------------------------
with tabs[2]:
    require_data()
    port_ret = st.session_state["port_ret"]

    st.subheader("VaR Summary")
    summary = var_summary(port_ret, confidence_levels=[0.95, 0.99])
    st.dataframe(summary.style.format("{:.4%}"))

    st.subheader("VaR + Tail Risk (LEL / WLEL) — BSc Dissertation Extension")
    tail_summary = var_summary_with_tail_risk(port_ret, confidence_levels=[0.95, 0.99])
    st.dataframe(tail_summary.style.format("{:.4%}"))
    st.caption(
        "LEL: average loss beyond the VaR threshold. WLEL: the same, with extreme "
        "losses up-weighted — following Basak & Shapiro (2001) and Chen & Nguyen (2024), "
        "as extended in my BSc dissertation."
    )

# ---------------------------------------------------------------------------
# TAB 4 — Backtesting
# ---------------------------------------------------------------------------
with tabs[3]:
    require_data()
    port_ret = st.session_state["port_ret"]

    col_a, col_b = st.columns(2)
    with col_a:
        confidence_label = st.radio("Confidence level", ["95%", "99%"], horizontal=True)
        confidence = 0.95 if confidence_label == "95%" else 0.99
    with col_b:
        window = st.slider("Rolling VaR window (days)", 60, 500, 250)

    st.caption(f"Testing VaR at **{confidence:.0%}** confidence, using a **{window}-day** rolling window.")

    with st.spinner("Computing rolling VaR and running backtests..."):
        alpha = 1 - confidence
        rolling_var = port_ret.rolling(window).apply(lambda x: -np.percentile(x, alpha * 100))
        exceptions = compute_exceptions(port_ret, rolling_var).dropna()
        kupiec = kupiec_test(exceptions, confidence)
        christoffersen = christoffersen_test(exceptions)

    m1, m2, m3 = st.columns(3)
    m1.metric("Observations tested", kupiec["n_observations"])
    m2.metric("Exceptions (breaches)", kupiec["n_exceptions"])
    m3.metric("Observed breach rate", f"{kupiec['observed_rate']:.2%}", f"expected {kupiec['expected_rate']:.2%}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Kupiec Test (Proportion of Failures)")
        st.json(kupiec)
    with col2:
        st.subheader("Christoffersen Test (Independence)")
        st.json(christoffersen)

# ---------------------------------------------------------------------------
# TAB 5 — Stress Testing
# ---------------------------------------------------------------------------
with tabs[4]:
    require_data()
    weights = st.session_state["weights"]

    st.subheader("Scenario Losses")
    results = run_all_scenarios(weights)
    st.dataframe(results.style.format({"portfolio_loss": "{:.2%}"}))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(results["scenario"], results["portfolio_loss"])
    ax.set_xlabel("Portfolio loss")
    st.pyplot(fig)

# ---------------------------------------------------------------------------
# TAB 6 — Climate VaR
# ---------------------------------------------------------------------------
with tabs[5]:
    require_data()
    returns = st.session_state["returns"]

    if len(returns.columns) < 2:
        st.warning("Load at least 2 tickers (one as market proxy) to run Climate VaR.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            asset_ticker = st.selectbox("Asset", returns.columns, index=len(returns.columns) - 1)
        with col2:
            market_ticker = st.selectbox("Market proxy", returns.columns, index=0)

        if st.button("Run Climate VaR"):
            with st.spinner("Estimating CAPM beta and running NGFS scenarios..."):
                beta = estimate_capm_beta(returns[asset_ticker], returns[market_ticker])
                st.metric("CAPM Beta", f"{beta:.3f}")

                climate_table = run_climate_var_scenarios(
                    returns[asset_ticker], returns[market_ticker],
                    scenarios=SAMPLE_NGFS_SCENARIOS, n_simulations=10_000,
                )
                st.dataframe(climate_table.style.format("{:.2f}"))
                st.session_state["climate_beta"] = beta

        st.caption(
            "Real NGFS Phase 5 data (GCAM 6.0 model, World GDP|MER, "
            "Counterfactual without damage) — see the Data Notes section in "
            "the README for the full methodology and source."
        )

# ---------------------------------------------------------------------------
# TAB 7 — Hybrid VaR
# ---------------------------------------------------------------------------
with tabs[6]:
    require_data()
    returns = st.session_state["returns"]

    if "climate_beta" not in st.session_state:
        st.info("Run the Climate VaR tab first to estimate a beta.")
    else:
        asset_ticker = st.selectbox("Asset for Hybrid VaR", returns.columns, index=len(returns.columns) - 1, key="hybrid_asset")
        climate_prob = st.slider("Climate-shock regime probability", 0.0, 0.5, 0.15, 0.05)
        target_horizon = st.slider("Target horizon (days)", 1, 20, 5)

        if st.button("Run Hybrid VaR"):
            with st.spinner("Running regime-switching Monte Carlo..."):
                beta = st.session_state["climate_beta"]
                scenario_shocks = {
                    name: climate_scenario_shock(chg, beta)
                    for name, chg in SAMPLE_NGFS_SCENARIOS.items()
                }
                hybrid_table = run_hybrid_var_scenarios(
                    returns[asset_ticker], scenario_shocks,
                    target_horizon_days=target_horizon, climate_prob=climate_prob,
                    n_simulations=10_000,
                )
                st.dataframe(hybrid_table.style.format("{:.2f}"))
        st.caption(
            "Regime-switching mechanic from my BSc dissertation's Hybrid Market-and-Climate "
            "VaR — each simulated day is drawn from either the baseline market regime or a "
            "climate-shocked regime, blended by probability."
        )

# ---------------------------------------------------------------------------
# TAB 8 — Earnings Event Overlay (Fusion)
# ---------------------------------------------------------------------------
with tabs[7]:
    require_data()
    port_ret = st.session_state["port_ret"]

    st.subheader("Live Earnings Disclosure Scoring")
    st.caption(
        "Paste any earnings-release text below to score its sentiment and see the "
        "resulting event-conditional VaR shift — the fusion of my BSc regime-switching "
        "mechanic and MSc dissertation's disclosure-scoring approach."
    )

    default_text = (
        "We delivered record revenue growth this quarter, with strong margin expansion "
        "and robust momentum across all segments. Guidance is being raised given our "
        "confidence in continued growth and resilience."
    )
    disclosure_text = st.text_area("Earnings disclosure text", value=default_text, height=150)

    if st.button("Score & Compute Event VaR"):
        scorer = EarningsSignalScorer()
        breakdown = scorer.score_breakdown(disclosure_text)

        col1, col2, col3 = st.columns(3)
        col1.metric("Sentiment Score", f"{breakdown['score']:.3f}", help="Range roughly [-1, 1]")
        col2.metric("Positive words", breakdown["counts"]["positive"])
        col3.metric("Negative words", breakdown["counts"]["negative"])

        event_result = event_conditional_var(
            baseline_mean=port_ret.mean(), baseline_std=port_ret.std(),
            llm_score=breakdown["score"],
        )
        st.subheader("Event-Conditional VaR")
        st.dataframe(pd.DataFrame([event_result]).style.format("{:.4f}"))

    st.divider()
    st.caption(
        "⚠️ This scorer uses a free, local lexicon-based proxy (Loughran-McDonald-style "
        "categories) — not a real LLM call. See src/earnings_signal.py's LLMSignalScorer "
        "stub for the upgrade path. Framed throughout as a risk-model validation exercise, "
        "not an autonomous trading signal — consistent with both source dissertations' own "
        "conclusions."
    )

# ---------------------------------------------------------------------------
# TAB 9 — ML vs GARCH
# ---------------------------------------------------------------------------
with tabs[8]:
    require_data()
    port_ret = st.session_state["port_ret"]

    if st.button("Run ML vs GARCH Comparison"):
        with st.spinner("Fitting GARCH, Random Forest and XGBoost..."):
            garch_fit = fit_garch(port_ret)
            garch_vol = pd.Series(garch_fit.conditional_volatility / 100, index=port_ret.index)
            comparison = compare_ml_vs_garch(port_ret, garch_vol, test_size=0.25)

        st.subheader("Out-of-Sample Forecast Accuracy (lower is better)")
        st.dataframe(comparison.style.format("{:.6f}"))

        winner = comparison["QLIKE"].idxmin()
        st.success(f"Best QLIKE (the metric that matters most for risk management): **{winner}**")

        X = build_features(port_ret)
        y = build_target(port_ret)
        X_train, X_test, y_train, y_test = train_test_split_time_series(X, y)
        rf_model = fit_random_forest(X_train, y_train)
        importances = feature_importance_report(rf_model, X_train.columns.tolist())

        st.subheader("Feature Importances (Random Forest)")
        fig, ax = plt.subplots(figsize=(7, 5))
        importances.head(10).plot(kind="barh", ax=ax)
        ax.invert_yaxis()
        st.pyplot(fig)

    st.caption(
        "Deliberately built as a genuine open question, not an assumed ML win — "
        "if GARCH wins, that's the correct and more credible finding to report."
    )
