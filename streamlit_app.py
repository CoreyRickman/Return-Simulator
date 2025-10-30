"""
Streamlit App: PPLI v1.1 (More Realistic)
-------------------------------------------------
Run locally:
  pip install -r requirements.txt
  streamlit run streamlit_app.py

requirements.txt (create this file):
  streamlit>=1.36
  pandas>=2.0
  numpy>=1.24

Notes:
- Implements the v1.1 spec: event-based tax approximation, annual rebalancing (with rebalancing turnover factor φ),
  monthly contributions, PPLI premium load and asset charge.
- Taxes are computed annually using aggregated realized components; monthly compounding is used for cashflow timing.
- Loss carryforward is handled annually across the taxable sleeve.
- One engine function drives all scenarios.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
import streamlit as st

Asset = Literal["stocks", "bonds", "alts"]
Wrapper = Literal["taxable", "ppli"]
Style = Literal["buy_and_hold", "rebalanced"]

# -----------------------------
# Data Models
# -----------------------------
@dataclass
class AssetParams:
    total: float         # expected total annual return
    income_share: float  # fraction of total that is income

@dataclass
class ScenarioConfig:
    name: str
    wrapper: Wrapper
    enhanced: bool

@dataclass
class InputConfig:
    starting_capital: float
    horizon_years: int
    weights: Dict[Asset, float]
    tilts: Dict[Asset, float]
    expected_returns: Dict[Asset, AssetParams]
    turnover: Dict[Asset, float]
    tax_rates: Dict[str, float]  # keys: ordinary, cap_gains, niit, state (optional)
    wrapper_costs: Dict[Wrapper, Dict[str, float]]
    cashflow: Dict[str, object]
    strategy_styles: List[Style]
    scenarios: List[ScenarioConfig]
    rebalance_phi: float = 0.20  # default rebalancing turnover factor for taxable

# -----------------------------
# Utility helpers
# -----------------------------

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def normalize_weights(w: Dict[Asset, float]) -> Dict[Asset, float]:
    total = sum(max(0.0, v) for v in w.values())
    if total <= 0:
        # fallback equal weights
        n = len(w)
        return {k: 1.0 / n for k in w}
    return {k: max(0.0, v) / total for k, v in w.items()}


def income_rate(asset: Asset, params: AssetParams) -> float:
    return params.total * clamp01(params.income_share)


def price_rate(asset: Asset, params: AssetParams) -> float:
    return params.total * (1.0 - clamp01(params.income_share))


def realized_price_rate(asset: Asset, params: AssetParams, turnover: float) -> float:
    return price_rate(asset, params) * clamp01(turnover)


def unrealized_price_rate(asset: Asset, params: AssetParams, turnover: float) -> float:
    return price_rate(asset, params) - realized_price_rate(asset, params, turnover)


def tax_rate_income(asset: Asset, tr: Dict[str, float]) -> float:
    ordinary = tr.get("ordinary", 0.0)
    cap = tr.get("cap_gains", 0.0)
    niit = tr.get("niit", 0.0)
    state = tr.get("state", 0.0)
    # Defaults: stocks income at cap gains; bonds at ordinary; alts 50/50 blend
    if asset == "stocks":
        return cap + niit + state
    elif asset == "bonds":
        return ordinary + niit + state
    else:  # alts
        return 0.5 * (ordinary + niit + state) + 0.5 * (cap + niit + state)


def tax_rate_realized(tr: Dict[str, float]) -> float:
    cap = tr.get("cap_gains", 0.0)
    niit = tr.get("niit", 0.0)
    state = tr.get("state", 0.0)
    return cap + niit + state


# -----------------------------
# Core Engine
# -----------------------------

def run_scenario(
    cfg: InputConfig,
    scenario: ScenarioConfig,
    style: Style,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run one scenario×style. Returns (timeseries df, KPIs dict).

    Timeseries columns:
      year, total_net, total_gross, drag, wrapper, style, scenario
      plus sleeve balances: stocks, bonds, alts
    """
    N = int(cfg.horizon_years)
    tr = cfg.tax_rates

    # Apply tilts if enhanced
    base_w = dict(cfg.weights)
    if scenario.enhanced:
        tw = {a: base_w[a] + cfg.tilts.get(a, 0.0) for a in base_w}
        w_target = normalize_weights(tw)  # re-normalize to sum 1
    else:
        w_target = normalize_weights(base_w)

    # Initialize balances by target weights (single wrapper per scenario)
    balances = {a: cfg.starting_capital * w_target[a] for a in w_target}
    gross_balances = balances.copy()  # parallel path with no taxes/fees

    # Loss carryforward for taxable wrapper
    loss_cf = 0.0

    # Contributions
    contribs = cfg.cashflow.get("contributions", []) or []
    contrib_by_year = {int(d.get("year", 0)): float(d.get("amount", 0.0)) for d in contribs}
    timing = (cfg.cashflow.get("timing") or "monthly").lower()

    # Wrapper costs
    if scenario.wrapper == "ppli":
        prem_load = float(cfg.wrapper_costs["ppli"].get("premium_load", 0.0))
        asset_chg = float(cfg.wrapper_costs["ppli"].get("asset_charge", 0.0))
        admin_flat = float(cfg.wrapper_costs["ppli"].get("admin_flat", 0.0))
    else:
        prem_load = 0.0
        asset_chg = float(cfg.wrapper_costs["taxable"].get("asset_charge", 0.0))
        admin_flat = 0.0

    rows = []

    for year in range(1, N + 1):
        # 1) Determine year contribution
        c_year = float(contrib_by_year.get(year, 0.0))
        if timing == "start_of_year":
            c_parts = [c_year]  # single chunk upfront
            parts = 1
        elif timing == "end_of_year":
            c_parts = [0.0]  # grow first, add at end
            parts = 1
        else:  # monthly default
            parts = 12
            c_parts = [c_year / 12.0] * 12

        # Accumulators for realized components (for annual taxes in taxable)
        realized_income = {a: 0.0 for a in balances}
        realized_price = {a: 0.0 for a in balances}
        unrealized_price_amt = {a: 0.0 for a in balances}

        # 2) Iterate periods within the year
        for i in range(parts):
            # Allocate contribution to sleeves by target weights
            c_i = c_parts[i]
            if scenario.wrapper == "ppli":
                # Apply premium load upon entry
                net_c = c_i * (1.0 - prem_load)
            else:
                net_c = c_i

            for a in balances:
                balances[a] += net_c * w_target[a]
                gross_balances[a] += c_i * w_target[a]

            # Compute monthly/periodic returns
            # Annual rates broken into monthly equivalents for compounding
            for a in balances:
                ap = cfg.expected_returns[a]
                inc_r = income_rate(a, ap)
                rp_r = realized_price_rate(a, ap, cfg.turnover[a])
                up_r = unrealized_price_rate(a, ap, cfg.turnover[a])

                r_inc_m = inc_r / parts
                r_rp_m = rp_r / parts
                r_up_m = up_r / parts

                if scenario.wrapper == "taxable":
                    # Accrue components before tax
                    base = balances[a]
                    inc_amt = base * r_inc_m
                    rp_amt = base * r_rp_m
                    up_amt = base * r_up_m

                    # Track components
                    realized_income[a] += inc_amt
                    realized_price[a] += rp_amt
                    unrealized_price_amt[a] += up_amt

                    # Apply growth net of nothing yet (taxes later annually)
                    balances[a] += inc_amt + rp_amt + up_amt
                else:
                    # PPLI: apply asset charge pro-rata, no taxes
                    base = balances[a]
                    gross_gain = base * (r_inc_m + r_rp_m + r_up_m)
                    fee = base * (asset_chg / parts)
                    balances[a] += gross_gain - fee

                # Gross path (no taxes, no fees except contributions not loaded)
                gb_base = gross_balances[a]
                gb_gain = gb_base * (r_inc_m + r_rp_m + r_up_m)
                # For gross path inside PPLI, exclude wrapper fees and premium load
                gross_balances[a] += gb_gain

        # 3) End-of-year rebalancing
        if style == "rebalanced":
            total_val = sum(balances.values())
            if total_val > 0:
                # Determine sales/buys needed; approximate additional realized gains in taxable
                for a in balances:
                    target_val = total_val * w_target[a]
                    diff = balances[a] - target_val
                    if scenario.wrapper == "taxable" and diff > 0:
                        # Selling winners: realize a fraction φ of the year’s unrealized gain
                        up_amt = unrealized_price_amt[a]
                        realized_extra = max(0.0, up_amt) * float(cfg.rebalance_phi)
                        realized_price[a] += realized_extra
                        balances[a] -= realized_extra * tax_rate_realized(tr)  # remove tax cash effect later (simplified)
                # Now actually set to targets exactly (no transaction costs modeled)
                total_val = sum(balances.values())
                for a in balances:
                    balances[a] = total_val * w_target[a]
                # Gross path also rebalanced without tax effects
                total_gross = sum(gross_balances.values())
                for a in gross_balances:
                    gross_balances[a] = total_gross * w_target[a]

        # 4) Annual taxes for taxable wrapper
        taxes_income = 0.0
        taxes_gains = 0.0
        if scenario.wrapper == "taxable":
            # Income taxes per asset
            for a in balances:
                rate_inc = tax_rate_income(a, tr)
                taxes_income += realized_income[a] * rate_inc

            # Capital gains tax with loss carryforward
            cap_rate = tax_rate_realized(tr)
            total_realized_gains = sum(max(0.0, realized_price[a]) for a in balances)
            # Apply loss carryforward
            net_gains = total_realized_gains + loss_cf
            if net_gains >= 0:
                taxes_gains = net_gains * cap_rate
                loss_cf = 0.0
            else:
                # Still negative → carry forward
                taxes_gains = 0.0
                loss_cf = net_gains  # negative value

            # Deduct taxes from balances pro-rata by sleeve value
            total_val = sum(balances.values())
            if total_val > 0:
                for a in balances:
                    share = balances[a] / total_val
                    balances[a] -= share * (taxes_income + taxes_gains)

        # 5) End-of-year end-timing contribution (if selected)
        if timing == "end_of_year" and c_year > 0:
            add = c_year * (1.0 - prem_load) if scenario.wrapper == "ppli" else c_year
            for a in balances:
                balances[a] += add * w_target[a]
                gross_balances[a] += c_year * w_target[a]

        # 6) Admin flat (PPLI)
        if scenario.wrapper == "ppli" and admin_flat > 0:
            total_val = sum(balances.values())
            if total_val > 0:
                for a in balances:
                    share = balances[a] / total_val
                    balances[a] -= share * admin_flat

        # 7) Record row
        total_net = sum(balances.values())
        total_gross = sum(gross_balances.values())
        drag = total_gross - total_net
        row = {
            "year": year,
            "total_net": total_net,
            "total_gross": total_gross,
            "drag": drag,
            "wrapper": scenario.wrapper,
            "style": style,
            "scenario": scenario.name,
        }
        for a in balances:
            row[a] = balances[a]
        rows.append(row)

    ts = pd.DataFrame(rows)

    # KPIs
    start_basis = cfg.starting_capital
    end_gross = float(ts["total_gross"].iloc[-1])
    end_net = float(ts["total_net"].iloc[-1])
    total_drag = end_gross - end_net
    years = float(N)
    # Annualized: guard against non-positive basis
    ann_gross = (end_gross / start_basis) ** (1.0 / years) - 1.0 if start_basis > 0 else float("nan")
    ann_net = (end_net / start_basis) ** (1.0 / years) - 1.0 if start_basis > 0 else float("nan")

    kpis = {
        "projected_value": end_net,
        "total_gross": end_gross,
        "total_net": end_net,
        "total_drag": total_drag,
        "ann_gross": ann_gross,
        "ann_net": ann_net,
    }

    return ts, kpis


# -----------------------------
# Streamlit UI
# -----------------------------

def default_config() -> InputConfig:
    return InputConfig(
        starting_capital=10_000_000.0,
        horizon_years=20,
        weights={"stocks": 0.70, "bonds": 0.20, "alts": 0.10},
        tilts={"stocks": 0.0, "bonds": 0.0, "alts": 0.0},
        expected_returns={
            "stocks": AssetParams(total=0.08, income_share=0.30),
            "bonds": AssetParams(total=0.05, income_share=0.90),
            "alts": AssetParams(total=0.08, income_share=0.50),
        },
        turnover={"stocks": 0.10, "bonds": 0.15, "alts": 0.25},
        tax_rates={"ordinary": 0.37, "cap_gains": 0.20, "niit": 0.038, "state": 0.0},
        wrapper_costs={
            "taxable": {"asset_charge": 0.0},
            "ppli": {"premium_load": 0.03, "asset_charge": 0.005, "admin_flat": 0.0},
        },
        cashflow={"contributions": [], "timing": "monthly"},
        strategy_styles=["buy_and_hold", "rebalanced"],
        scenarios=[
            ScenarioConfig(name="Traditional Taxable", wrapper="taxable", enhanced=False),
            ScenarioConfig(name="Traditional PPLI", wrapper="ppli", enhanced=False),
            ScenarioConfig(name="Alts Enhanced Tax", wrapper="taxable", enhanced=True),
            ScenarioConfig(name="Alts Enhanced PPLI", wrapper="ppli", enhanced=True),
        ],
        rebalance_phi=0.20,
    )


def sidebar_inputs(cfg: InputConfig) -> InputConfig:
    st.sidebar.header("Inputs")
    sc = float(st.sidebar.number_input("Starting Capital", min_value=0.0, value=cfg.starting_capital, step=100000.0, format="%.2f"))
    horizon = int(st.sidebar.number_input("Horizon (years)", min_value=1, value=cfg.horizon_years, step=1))

    st.sidebar.subheader("Weights")
    w_s = float(st.sidebar.number_input("Stocks weight", 0.0, 1.0, cfg.weights["stocks"], 0.01))
    w_b = float(st.sidebar.number_input("Bonds weight", 0.0, 1.0, cfg.weights["bonds"], 0.01))
    w_a = float(st.sidebar.number_input("Alts weight", 0.0, 1.0, cfg.weights["alts"], 0.01))

    st.sidebar.subheader("Tilts (for Enhanced scenarios)")
    t_s = float(st.sidebar.number_input("Stocks tilt", -1.0, 1.0, cfg.tilts["stocks"], 0.01))
    t_b = float(st.sidebar.number_input("Bonds tilt", -1.0, 1.0, cfg.tilts["bonds"], 0.01))
    t_a = float(st.sidebar.number_input("Alts tilt", -1.0, 1.0, cfg.tilts["alts"], 0.01))

    st.sidebar.subheader("Expected Returns")
    s_total = float(st.sidebar.number_input("Stocks total return", -1.0, 1.0, cfg.expected_returns["stocks"].total, 0.001))
    s_inc = float(st.sidebar.number_input("Stocks income share", 0.0, 1.0, cfg.expected_returns["stocks"].income_share, 0.01))
    b_total = float(st.sidebar.number_input("Bonds total return", -1.0, 1.0, cfg.expected_returns["bonds"].total, 0.001))
    b_inc = float(st.sidebar.number_input("Bonds income share", 0.0, 1.0, cfg.expected_returns["bonds"].income_share, 0.01))
    a_total = float(st.sidebar.number_input("Alts total return", -1.0, 1.0, cfg.expected_returns["alts"].total, 0.001))
    a_inc = float(st.sidebar.number_input("Alts income share", 0.0, 1.0, cfg.expected_returns["alts"].income_share, 0.01))

    st.sidebar.subheader("Turnover")
    to_s = float(st.sidebar.number_input("Stocks turnover", 0.0, 1.0, cfg.turnover["stocks"], 0.01))
    to_b = float(st.sidebar.number_input("Bonds turnover", 0.0, 1.0, cfg.turnover["bonds"], 0.01))
    to_a = float(st.sidebar.number_input("Alts turnover", 0.0, 1.0, cfg.turnover["alts"], 0.01))

    st.sidebar.subheader("Tax Rates")
    tr_o = float(st.sidebar.number_input("Ordinary", 0.0, 1.0, cfg.tax_rates["ordinary"], 0.001))
    tr_c = float(st.sidebar.number_input("Cap Gains", 0.0, 1.0, cfg.tax_rates["cap_gains"], 0.001))
    tr_n = float(st.sidebar.number_input("NIIT", 0.0, 1.0, cfg.tax_rates["niit"], 0.001))
    tr_s = float(st.sidebar.number_input("State", 0.0, 1.0, cfg.tax_rates.get("state", 0.0), 0.001))

    st.sidebar.subheader("PPLI Costs")
    pl = float(st.sidebar.number_input("Premium load", 0.0, 0.20, cfg.wrapper_costs["ppli"]["premium_load"], 0.001))
    ac = float(st.sidebar.number_input("Asset charge", 0.0, 0.05, cfg.wrapper_costs["ppli"]["asset_charge"], 0.0005))
    af = float(st.sidebar.number_input("Admin flat ($)", 0.0, 1_000_000.0, cfg.wrapper_costs["ppli"].get("admin_flat", 0.0), 100.0))

    st.sidebar.subheader("Contributions")
    timing = st.sidebar.selectbox("Timing", ["monthly", "start_of_year", "end_of_year"], index=["monthly", "start_of_year", "end_of_year"].index(cfg.cashflow.get("timing", "monthly")))

    # Contributions editor
    df_contrib = pd.DataFrame(cfg.cashflow.get("contributions", []) or [], columns=["year", "amount"]).astype({"year": int, "amount": float})
    st.sidebar.caption("Add rows for contribution schedule (year, amount)")
    df_contrib = st.sidebar.data_editor(df_contrib, num_rows="dynamic", use_container_width=True)

    st.sidebar.subheader("Advanced")
    phi = float(st.sidebar.number_input("Rebalancing turnover φ (taxable)", 0.0, 1.0, cfg.rebalance_phi, 0.05))

    # Build new config
    new_cfg = InputConfig(
        starting_capital=sc,
        horizon_years=horizon,
        weights=normalize_weights({"stocks": w_s, "bonds": w_b, "alts": w_a}),
        tilts={"stocks": t_s, "bonds": t_b, "alts": t_a},
        expected_returns={
            "stocks": AssetParams(total=s_total, income_share=s_inc),
            "bonds": AssetParams(total=b_total, income_share=b_inc),
            "alts": AssetParams(total=a_total, income_share=a_inc),
        },
        turnover={"stocks": to_s, "bonds": to_b, "alts": to_a},
        tax_rates={"ordinary": tr_o, "cap_gains": tr_c, "niit": tr_n, "state": tr_s},
        wrapper_costs={
            "taxable": {"asset_charge": 0.0},
            "ppli": {"premium_load": pl, "asset_charge": ac, "admin_flat": af},
        },
        cashflow={"contributions": df_contrib.to_dict(orient="records"), "timing": timing},
        strategy_styles=["buy_and_hold", "rebalanced"],
        scenarios=[
            ScenarioConfig(name="Traditional Taxable", wrapper="taxable", enhanced=False),
            ScenarioConfig(name="Traditional PPLI", wrapper="ppli", enhanced=False),
            ScenarioConfig(name="Alts Enhanced Tax", wrapper="taxable", enhanced=True),
            ScenarioConfig(name="Alts Enhanced PPLI", wrapper="ppli", enhanced=True),
        ],
        rebalance_phi=phi,
    )
    return new_cfg


def format_currency(x: float) -> str:
    try:
        return f"${x:,.0f}"
    except Exception:
        return str(x)


def app():
    st.set_page_config(page_title="PPLI Model v1.1", layout="wide")
    st.title("PPLI vs Taxable — v1.1 More Realistic")

    cfg = default_config()
    cfg = sidebar_inputs(cfg)

    st.markdown("### Scenarios")
    st.caption("Runs four scenarios (Traditional vs Alts Enhanced) across two wrappers (Taxable vs PPLI) and both Buy & Hold and Rebalanced styles.")

    # Run all
    results: List[pd.DataFrame] = []
    kpi_rows = []
    for sc in cfg.scenarios:
        for style in cfg.strategy_styles:
            ts, kpis = run_scenario(cfg, sc, style) 
            results.append(ts)
            kpi_rows.append({
                "Scenario": sc.name,
                "Wrapper": sc.wrapper,
                "Style": style,
                "Projected Value": kpis["projected_value"],
                "Total Gross": kpis["total_gross"],
                "Total Net": kpis["total_net"],
                "Total Drag": kpis["total_drag"],
                "Ann Gross": kpis["ann_gross"],
                "Ann Net": kpis["ann_net"],
            })

    all_ts = pd.concat(results, ignore_index=True)
    kpi_df = pd.DataFrame(kpi_rows)

    # Display KPIs table
    display = kpi_df.copy()
    for c in ["Projected Value", "Total Gross", "Total Net", "Total Drag"]:
        display[c] = display[c].apply(format_currency)
    for c in ["Ann Gross", "Ann Net"]:
        display[c] = (kpi_df[c] * 100).map(lambda v: f"{v:.2f}%")

    st.dataframe(display, use_container_width=True, hide_index=True)

    # Charts
    left, right = st.columns(2)
    with left:
        st.markdown("#### Wealth Over Time — Net")
        sel = st.selectbox("Select scenario for chart", options=sorted(all_ts["scenario"].unique()))
        chart_df = all_ts[all_ts["scenario"] == sel].pivot(index="year", columns=["wrapper", "style"], values="total_net")
        st.line_chart(chart_df)
    with right:
        st.markdown("#### Cumulative Drag (Gross − Net)")
        drag_df = all_ts[all_ts["scenario"] == sel].pivot(index="year", columns=["wrapper", "style"], values="drag")
        st.line_chart(drag_df)

    # Download
    st.markdown("### Download Results")
    dl_ts = all_ts.sort_values(["scenario", "wrapper", "style", "year"])  # tidy
    st.download_button(
        label="Download Time Series (CSV)",
        data=dl_ts.to_csv(index=False).encode("utf-8"),
        file_name="ppli_v11_timeseries.csv",
        mime="text/csv",
    )
    st.download_button(
        label="Download KPIs (CSV)",
        data=kpi_df.to_csv(index=False).encode("utf-8"),
        file_name="ppli_v11_kpis.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.caption("This is a v1.1 educational model. Taxes and policy costs are approximations. For institutional use, consider adding turnover-by-asset, short/long holding periods, and detailed carrier cost schedules.")


if __name__ == "__main__":
    app()
