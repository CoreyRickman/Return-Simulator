"""
Streamlit App: PPLI v1.1.1 (Safer, Cloud-ready)
-------------------------------------------------
- Implements the "more realistic" engine (event-based tax approximation, annual taxes, rebalancing tax via φ,
  monthly contributions, PPLI premium load + asset charge, loss carryforward).
- Hardens app flow for Streamlit Cloud (no MultiIndex charts, guards for empty results, deprecation updates).

Run locally:
  pip install -r requirements.txt
  streamlit run streamlit_app.py

Recommended requirements.txt:
  streamlit==1.51.0
  pandas==2.3.3
  numpy==2.3.4

Optional runtime.txt (for Streamlit Cloud):
  3.11
"""

from __future__ import annotations
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
    income_share: float  # fraction of total that is income (0..1)

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
    tax_rates: Dict[str, float]  # ordinary, cap_gains, niit, state
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
        n = len(w)
        return {k: 1.0 / n for k in w}
    return {k: max(0.0, v) / total for k, v in w.items()}


def income_rate(params: AssetParams) -> float:
    return params.total * clamp01(params.income_share)


def price_rate(params: AssetParams) -> float:
    return params.total * (1.0 - clamp01(params.income_share))


def realized_price_rate(params: AssetParams, turnover: float) -> float:
    return price_rate(params) * clamp01(turnover)


def unrealized_price_rate(params: AssetParams, turnover: float) -> float:
    return price_rate(params) - realized_price_rate(params, turnover)


def tax_rate_income(asset: Asset, tr: Dict[str, float]) -> float:
    ordinary = tr.get("ordinary", 0.0)
    cap = tr.get("cap_gains", 0.0)
    niit = tr.get("niit", 0.0)
    state = tr.get("state", 0.0)
    if asset == "stocks":
        # treat equity income as qualified (cap gains) by default
        return cap + niit + state
    if asset == "bonds":
        return ordinary + niit + state
    # alts blend 50/50
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

    Timeseries columns: year, total_net, total_gross, drag, wrapper, style, scenario, stocks, bonds, alts
    """
    N = int(cfg.horizon_years)
    tr = cfg.tax_rates

    # Target weights (apply tilts if enhanced)
    base_w = dict(cfg.weights)
    if scenario.enhanced:
        tw = {a: base_w[a] + cfg.tilts.get(a, 0.0) for a in base_w}
        w_target = normalize_weights(tw)
    else:
        w_target = normalize_weights(base_w)

    # Initialize balances by target weights
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

    rows: List[Dict[str, float]] = []

    for year in range(1, N + 1):
        # 1) Contribution schedule for the year
        c_year = float(contrib_by_year.get(year, 0.0))
        if timing == "start_of_year":
            parts = 1
            c_parts = [c_year]
        elif timing == "end_of_year":
            parts = 1
            c_parts = [0.0]  # add at end
        else:  # monthly default
            parts = 12
            c_parts = [c_year / 12.0] * 12

        realized_income = {a: 0.0 for a in balances}
        realized_price = {a: 0.0 for a in balances}
        unrealized_price_amt = {a: 0.0 for a in balances}

        for i in range(parts):
            c_i = c_parts[i]
            if scenario.wrapper == "ppli":
                net_c = c_i * (1.0 - prem_load)
            else:
                net_c = c_i

            for a in balances:
                balances[a] += net_c * w_target[a]
                gross_balances[a] += c_i * w_target[a]

            for a in balances:
                ap = cfg.expected_returns[a]
                inc_r = income_rate(ap) / parts
                rp_r = realized_price_rate(ap, cfg.turnover[a]) / parts
                up_r = unrealized_price_rate(ap, cfg.turnover[a]) / parts

                if scenario.wrapper == "taxable":
                    base = balances[a]
                    inc_amt = base * inc_r
                    rp_amt = base * rp_r
                    up_amt = base * up_r
                    realized_income[a] += inc_amt
                    realized_price[a] += rp_amt
                    unrealized_price_amt[a] += up_amt
                    balances[a] += inc_amt + rp_amt + up_amt
                else:
                    base = balances[a]
                    gross_gain = base * (inc_r + rp_r + up_r)
                    fee = base * (asset_chg / parts)
                    balances[a] += gross_gain - fee

                # Gross path (no taxes/fees; premium load is excluded from gross to match the meaning of "gross")
                gb_base = gross_balances[a]
                gb_gain = gb_base * (inc_r + rp_r + up_r)
                gross_balances[a] += gb_gain

        # 3) Rebalancing at year-end
        if style == "rebalanced":
            # Approximate rebalancing-induced realization (taxable only)
            if scenario.wrapper == "taxable":
                for a in balances:
                    # realize a fraction φ of this year's unrealized growth
                    up_amt = max(0.0, unrealized_price_amt[a])
                    realized_price[a] += up_amt * float(cfg.rebalance_phi)
                # NOTE: taxes will be applied in the annual tax block below; do NOT subtract here

            # Set balances to target weights (no transaction costs modeled)
            total_val = sum(balances.values())
            if total_val > 0:
                for a in balances:
                    balances[a] = total_val * w_target[a]
            total_gross = sum(gross_balances.values())
            if total_gross > 0:
                for a in gross_balances:
                    gross_balances[a] = total_gross * w_target[a]

        # 4) Annual taxes (taxable wrapper)
        if scenario.wrapper == "taxable":
            taxes_income = 0.0
            for a in balances:
                taxes_income += realized_income[a] * tax_rate_income(a, tr)

            total_realized_gains = sum(max(0.0, realized_price[a]) for a in balances)
            cap_rate = tax_rate_realized(tr)
            net_gains = total_realized_gains + loss_cf
            if net_gains >= 0:
                taxes_gains = net_gains * cap_rate
                loss_cf = 0.0
            else:
                taxes_gains = 0.0
                loss_cf = net_gains  # negative carryforward

            total_val = sum(balances.values())
            if total_val > 0:
                tax_total = taxes_income + taxes_gains
                for a in balances:
                    share = balances[a] / total_val
                    balances[a] -= share * tax_total

        # 5) End-of-year contribution (if timing == end_of_year)
        if timing == "end_of_year" and c_year > 0:
            add = c_year * (1.0 - prem_load) if scenario.wrapper == "ppli" else c_year
            for a in balances:
                balances[a] += add * w_target[a]
                gross_balances[a] += c_year * w_target[a]

        # 6) Admin flat fees (PPLI)
        if scenario.wrapper == "ppli" and admin_flat > 0:
            total_val = sum(balances.values())
            if total_val > 0:
                for a in balances:
                    share = balances[a] / total_val
                    balances[a] -= share * admin_flat

        # 7) Record row
        total_net = float(sum(balances.values()))
        total_gross = float(sum(gross_balances.values()))
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
            row[a] = float(balances[a])
        rows.append(row)

    ts = pd.DataFrame(rows)

    # KPIs (use starting capital as basis)
    start_basis = float(cfg.starting_capital)
    end_gross = float(ts["total_gross"].iloc[-1]) if not ts.empty else float("nan")
    end_net = float(ts["total_net"].iloc[-1]) if not ts.empty else float("nan")
    total_drag = end_gross - end_net if np.isfinite(end_gross) and np.isfinite(end_net) else float("nan")
    years = float(N)
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

    df_contrib = pd.DataFrame(cfg.cashflow.get("contributions", []) or [], columns=["year", "amount"]).astype({"year": int, "amount": float})
    st.sidebar.caption("Add rows for contribution schedule (year, amount)")
    df_contrib = st.sidebar.data_editor(df_contrib, num_rows="dynamic", width="stretch")

    st.sidebar.subheader("Advanced")
    phi = float(st.sidebar.number_input("Rebalancing turnover φ (taxable)", 0.0, 1.0, cfg.rebalance_phi, 0.05))

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
    st.set_page_config(page_title="PPLI Model v1.1.1", layout="wide")
    st.title("PPLI vs Taxable — v1.1.1 (Safer)")

    cfg = default_config()
    cfg = sidebar_inputs(cfg)

    st.markdown("### Scenarios")
    st.caption("Runs four scenarios (Traditional vs Alts Enhanced) across two wrappers (Taxable vs PPLI) and both Buy & Hold and Rebalanced styles.")

    # Run all scenarios with guards
    results: List[pd.DataFrame] = []
    kpi_rows: List[Dict[str, float]] = []

    for sc in cfg.scenarios:
        for style in cfg.strategy_styles:
            try:
                ts, kpis = run_scenario(cfg, sc, style)
                if ts is None or ts.empty:
                    st.warning(f"No data produced for {sc.name} ({style}). Check inputs.")
                else:
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
            except Exception as e:
                st.warning(f"Scenario {sc.name} ({style}) failed: {e}")

    if not results:
        st.error("No scenarios ran successfully — please adjust inputs.")
        st.stop()

    all_ts = pd.concat(results, ignore_index=True)
    kpi_df = pd.DataFrame(kpi_rows)

    # KPI table
    display = kpi_df.copy()
    for c in ["Projected Value", "Total Gross", "Total Net", "Total Drag"]:
        display[c] = display[c].apply(format_currency)
    for c in ["Ann Gross", "Ann Net"]:
        display[c] = (kpi_df[c] * 100).map(lambda v: f"{v:.2f}%")

    st.dataframe(display, width="stretch", hide_index=True)

    # ------- Charts (flattened, no MultiIndex) -------
    def build_flat_pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        tmp = df.copy()
        tmp["series"] = (
            tmp["wrapper"].astype(str).str.upper()
            + " · "
            + tmp["style"].astype(str).str.replace("_", " ").str.title()
        )
        out = tmp.pivot(index="year", columns="series", values=value_col).sort_index()
        return out.apply(pd.to_numeric, errors="coerce")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Wealth Over Time — Net")
        options = sorted(all_ts["scenario"].unique()) if not all_ts.empty else []
        sel = st.selectbox("Select scenario for chart", options=options)
        scen = all_ts[all_ts["scenario"] == sel] if sel else pd.DataFrame()
        chart_df = build_flat_pivot(scen, value_col="total_net")
        if chart_df.empty:
            st.info("No data available for the selected scenario.")
        else:
            st.line_chart(chart_df)

    with right:
        st.markdown("#### Cumulative Drag (Gross − Net)")
        drag_df = build_flat_pivot(scen, value_col="drag")
        if drag_df.empty:
            st.info("No data available for the selected scenario.")
        else:
            st.line_chart(drag_df)

    # Downloads
    st.markdown("### Download Results")
    dl_ts = all_ts.sort_values(["scenario", "wrapper", "style", "year"])  # tidy
    st.download_button(
        label="Download Time Series (CSV)",
        data=dl_ts.to_csv(index=False).encode("utf-8"),
        file_name="ppli_v111_timeseries.csv",
        mime="text/csv",
    )
    st.download_button(
        label="Download KPIs (CSV)",
        data=kpi_df.to_csv(index=False).encode("utf-8"),
        file_name="ppli_v111_kpis.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.caption("Educational model. Taxes & policy costs are approximations; not tax advice.")


if __name__ == "__main__":
    app()
