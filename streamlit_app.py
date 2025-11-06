import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from typing import Dict, List, Tuple

st.set_page_config(page_title="PPLI vs Taxable Return Model", layout="wide")

# -----------------------------
# Constants & helpers
# -----------------------------
ASSETS = ["Stocks", "Bonds", "Alternatives"]
STRATEGIES = {
    "Buy & Hold (no annual rebalance)": "buy_hold",
    "Rebalanced annually to target weights": "rebalanced",
}

@st.cache_data
def _years_list(T: int):
    return list(range(0, T + 1))

def normalize_percents(values: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(v, 0.0) for v in values.values())
    if total == 0:
        first = list(values.keys())[0]
        return {k: (100.0 if k == first else 0.0) for k in values.keys()}
    return {k: 100.0 * max(v, 0.0) / total for k, v in values.items()}

def pct_to_decimal(x: float) -> float:
    return float(x) / 100.0

def annual_cagr(v0: float, vt: float, T: int) -> float:
    if T <= 0 or v0 <= 0:
        return 0.0
    return (vt / v0) ** (1.0 / T) - 1.0

# -----------------------------
# Simulation engine
# -----------------------------
def simulate_scenario(
    initial: float,
    years: int,
    strategy: str,
    asset_weights_pct: Dict[str, float],  # must sum to 100
    asset_returns_pct: Dict[str, float],  # GLOBAL
    ppli_cost_pct: float,                 # GLOBAL
    sleeve_pct_in_ppli: Dict[str, float],
    tax_ordinary_pct: float,              # GLOBAL
    tax_cg_pct: float,                    # GLOBAL
    tax_mix_pct: Dict[str, Tuple[float, float]],  # GLOBAL per asset (OI%, CG%)
):
    """
    Returns (path_gross, path_net) each as DataFrame with columns per-asset & Total.
    Notes:
    - Gross ignores taxes and PPLI cost.
    - Net applies PPLI cost to PPLI sleeve and blended tax to non-PPLI sleeve, annually.
    - Buy & Hold tracks separate sleeves (PPLI vs Taxable) per asset without rebalancing.
    - Rebalanced resets sleeves to target percentages at the start of each year.
    """
    # Normalize weights and tax mixes
    w = normalize_percents(asset_weights_pct)
    sleeve_ppli = {a: min(max(sleeve_pct_in_ppli.get(a, 0.0), 0.0), 100.0) for a in ASSETS}
    tax_mix = {
        a: normalize_percents({"OI": tax_mix_pct.get(a, (0.0, 100.0))[0], "CG": tax_mix_pct.get(a, (0.0, 100.0))[1]})
        for a in ASSETS
    }

    r_gross = {a: pct_to_decimal(asset_returns_pct.get(a, 0.0)) for a in ASSETS}
    tax_oi = pct_to_decimal(tax_ordinary_pct)
    tax_cg = pct_to_decimal(tax_cg_pct)
    ppli_cost = pct_to_decimal(ppli_cost_pct)

    # Precompute per-asset net return rates for each sleeve
    r_taxable_net = {}
    r_ppli_net = {}
    for a in ASSETS:
        mix_oi = pct_to_decimal(tax_mix[a]["OI"])
        mix_cg = pct_to_decimal(tax_mix[a]["CG"])
        blend_tax = mix_oi * tax_oi + mix_cg * tax_cg
        r_taxable_net[a] = r_gross[a] * (1.0 - blend_tax)
        r_ppli_net[a] = r_gross[a] - ppli_cost

    years_list = _years_list(years)

    # Initialize holdings for gross and net
    gross_values = {a: [initial * pct_to_decimal(w[a])] for a in ASSETS}
    net_taxable_values = {a: [initial * pct_to_decimal(w[a]) * (1.0 - pct_to_decimal(sleeve_ppli[a]))] for a in ASSETS}
    net_ppli_values = {a: [initial * pct_to_decimal(w[a]) * pct_to_decimal(sleeve_ppli[a])] for a in ASSETS}

    # Simulation loop
    for _ in years_list[1:]:
        if strategy == "rebalanced":
            # Gross rebalanced to target weights
            total_gross_prev = sum(gross_values[a][-1] for a in ASSETS)
            for a in ASSETS:
                gross_values[a][-1] = total_gross_prev * pct_to_decimal(w[a])
            # Net sleeves rebalanced to target sleeves per asset
            total_net_prev = sum(net_taxable_values[a][-1] + net_ppli_values[a][-1] for a in ASSETS)
            for a in ASSETS:
                target_asset_total = total_net_prev * pct_to_decimal(w[a])
                net_ppli_values[a][-1] = target_asset_total * pct_to_decimal(sleeve_ppli[a])
                net_taxable_values[a][-1] = target_asset_total - net_ppli_values[a][-1]

        # Grow for one year
        for a in ASSETS:
            gross_values[a].append(gross_values[a][-1] * (1.0 + r_gross[a]))
            net_taxable_values[a].append(net_taxable_values[a][-1] * (1.0 + r_taxable_net[a]))
            net_ppli_values[a].append(net_ppli_values[a][-1] * (1.0 + r_ppli_net[a]))

    # Build DataFrames
    gross_df = pd.DataFrame({a: gross_values[a] for a in ASSETS}, index=years_list)
    gross_df.index.name = "Year"
    gross_df["Total"] = gross_df.sum(axis=1)

    net_df = pd.DataFrame({a: np.array(net_taxable_values[a]) + np.array(net_ppli_values[a]) for a in ASSETS}, index=years_list)
    net_df.index.name = "Year"
    net_df["Total"] = net_df.sum(axis=1)

    return gross_df, net_df

def summarize_metrics(initial: float, gross_df: pd.DataFrame, net_df: pd.DataFrame) -> Dict[str, float]:
    vt_g = float(gross_df["Total"].iloc[-1])
    vt_n = float(net_df["Total"].iloc[-1])
    T = int(gross_df.shape[0] - 1)

    tr_g = vt_g / initial - 1.0
    tr_n = vt_n / initial - 1.0
    tr_drag = tr_g - tr_n

    ar_g = annual_cagr(initial, vt_g, T)
    ar_n = annual_cagr(initial, vt_n, T)
    ar_drag = ar_g - ar_n

    return {
        "End Gross Value": vt_g,
        "End Net Value": vt_n,
        "Total Return (Gross)": tr_g,
        "Total Return (Net)": tr_n,
        "Total Return Tax Drag": tr_drag,
        "Annualized (Gross)": ar_g,
        "Annualized (Net)": ar_n,
        "Annualized Tax Drag": ar_drag,
    }

# -----------------------------
# UI: Global inputs
# -----------------------------
st.title("📈 PPLI vs Taxable: Multi-Scenario Return Model (v1.5)")

with st.sidebar:
    st.header("Global Settings")

    initial = st.number_input("Initial Investment ($)", min_value=1000.0, value=1_000_000.0, step=1000.0, format="%0.2f")
    years = st.number_input("Investment Horizon (years)", min_value=1, max_value=100, value=20, step=1)

    st.subheader("Expected Annual Returns (% – before tax & PPLI)")
    c1, c2, c3 = st.columns(3)
    r_stocks = c1.number_input("Stocks r% (global)", min_value=-50.0, max_value=50.0, value=7.0, step=0.1)
    r_bonds  = c2.number_input("Bonds r% (global)",  min_value=-50.0, max_value=50.0, value=4.0, step=0.1)
    r_alts   = c3.number_input("Alternatives r% (global)", min_value=-50.0, max_value=50.0, value=6.0, step=0.1)
    returns_global = {"Stocks": r_stocks, "Bonds": r_bonds, "Alternatives": r_alts}

    st.subheader("Tax Assumptions (% per year – global)")
    # switched from sliders -> number_input for consistency
    t1, t2 = st.columns(2)
    tax_ordinary_pct = t1.number_input("Ordinary Income Tax %", min_value=0.0, max_value=70.0, value=37.0, step=0.1)
    tax_cg_pct       = t2.number_input("Capital Gains Tax %",  min_value=0.0, max_value=70.0, value=23.8, step=0.1)

    st.markdown("**Tax Mix of Returns (%, per asset – global)** — portion taxed as OI vs CG (normalized to 100%)")
    tm1, tm2, tm3 = st.columns(3)
    st_oi = tm1.number_input("Stocks % Ordinary (global)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
    st_cg = tm1.number_input("Stocks % Cap Gains (global)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
    bo_oi = tm2.number_input("Bonds % Ordinary (global)", min_value=0.0, max_value=100.0, value=100.0, step=1.0)
    bo_cg = tm2.number_input("Bonds % Cap Gains (global)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
    al_oi = tm3.number_input("Alts % Ordinary (global)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
    al_cg = tm3.number_input("Alts % Cap Gains (global)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
    tax_mix_global = {"Stocks": (st_oi, st_cg), "Bonds": (bo_oi, bo_cg), "Alternatives": (al_oi, al_cg)}

    st.subheader("PPLI")
    # switched from slider -> number_input
    ppli_cost_pct = st.number_input("PPLI Annual Cost % (global)", min_value=0.0, max_value=5.0, value=1.0, step=0.05)

    st.subheader("Strategy (global)")
    strategy_label = st.selectbox("Rebalancing", list(STRATEGIES.keys()), index=1)
    strategy_global = STRATEGIES[strategy_label]

    st.caption("Global settings apply to all scenarios. Scenarios only define allocations, PPLI sleeve, and custom names.")

# -----------------------------
# Scenario builder (weights + PPLI sleeve + names)
# -----------------------------
st.markdown("## Scenarios (choose allocations, PPLI sleeve, and names)")

scenario_tabs = st.tabs([f"Scenario {i+1}" for i in range(4)])
scenario_inputs = []

for i, tab in enumerate(scenario_tabs):
    with tab:
        name = st.text_input(f"Scenario {i+1} Name", value=f"Scenario {i+1}", key=f"name_{i}")

        st.markdown("**Portfolio Allocation (%)** — must total ~100%")
        w1, w2, w3 = st.columns(3)
        w_stocks = w1.number_input(f"Stocks % (S{i+1})", min_value=0.0, max_value=100.0, value=60.0, step=1.0, key=f"w_stocks_{i}")
        w_bonds  = w2.number_input(f"Bonds % (S{i+1})",  min_value=0.0, max_value=100.0, value=30.0, step=1.0, key=f"w_bonds_{i}")
        w_alts   = w3.number_input(f"Alternatives % (S{i+1})", min_value=0.0, max_value=100.0, value=10.0, step=1.0, key=f"w_alts_{i}")
        weights = {"Stocks": w_stocks, "Bonds": w_bonds, "Alternatives": w_alts}
        if abs(sum(weights.values()) - 100.0) > 0.01:
            st.warning(f"Allocations sum to {sum(weights.values()):.2f}%. They will be normalized to 100% for calculations.")

        st.markdown("**PPLI Sleeve (% of each asset held in PPLI)**")
        p1, p2, p3 = st.columns(3)
        p_st = p1.number_input(f"Stocks in PPLI % (S{i+1})", min_value=0.0, max_value=100.0, value=50.0, step=1.0, key=f"p_st_{i}")
        p_bo = p2.number_input(f"Bonds in PPLI % (S{i+1})",  min_value=0.0, max_value=100.0, value=0.0,  step=1.0, key=f"p_bo_{i}")
        p_al = p3.number_input(f"Alts in PPLI % (S{i+1})",   min_value=0.0, max_value=100.0, value=50.0, step=1.0, key=f"p_al_{i}")
        sleeve = {"Stocks": p_st, "Bonds": p_bo, "Alternatives": p_al}

        scenario_inputs.append({"name": name, "weights": weights, "sleeve": sleeve})

# -----------------------------
# Run simulations
# -----------------------------
results = []
series_net, series_gross = {}, {}

for sc in scenario_inputs:
    gross_df, net_df = simulate_scenario(
        initial=initial,
        years=years,
        strategy=strategy_global,
        asset_weights_pct=sc["weights"],
        asset_returns_pct=returns_global,
        ppli_cost_pct=ppli_cost_pct,
        sleeve_pct_in_ppli=sc["sleeve"],
        tax_ordinary_pct=tax_ordinary_pct,
        tax_cg_pct=tax_cg_pct,
        tax_mix_pct=tax_mix_global,
    )
    metrics = summarize_metrics(initial, gross_df, net_df)
    results.append({"Scenario": sc["name"], **metrics})
    series_net[f"{sc['name']} (Net)"] = net_df["Total"]
    series_gross[f"{sc['name']} (Gross)"] = gross_df["Total"]

# -----------------------------
# Outputs: Ranked table & charts
# -----------------------------
st.markdown("## Results")

metrics_df = pd.DataFrame(results)

# Rank scenarios by End Net Value (descending)
ranked_df = metrics_df.sort_values("End Net Value", ascending=False).reset_index(drop=True)
ranked_df.insert(0, "Rank", ranked_df.index + 1)

# Display-friendly formatting
metrics_df_display = ranked_df.copy()
money_cols = ["End Gross Value", "End Net Value"]
percent_cols = [
    "Total Return (Gross)",
    "Total Return (Net)",
    "Total Return Tax Drag",
    "Annualized (Gross)",
    "Annualized (Net)",
    "Annualized Tax Drag",
]
for c in money_cols:
    metrics_df_display[c] = metrics_df_display[c].map(lambda x: f"${x:,.0f}")
for c in percent_cols:
    metrics_df_display[c] = metrics_df_display[c].map(lambda x: f"{x*100:,.2f}%")

st.dataframe(metrics_df_display, use_container_width=True)

# Growth over time chart
st.markdown("### Growth Over Time")
show_gross = st.checkbox("Show Gross lines (untaxed, no PPLI cost)", value=False)
plot_df = pd.DataFrame(index=_years_list(years))
for name, s in series_net.items():
    plot_df[name] = s
if show_gross:
    for name, s in series_gross.items():
        plot_df[name] = s

st.line_chart(plot_df, height=420)

# Tax efficiency scatter: Return vs Tax Drag
st.markdown("### Tax Efficiency: Return vs. Tax Drag")
mode = st.radio("Measure of return & drag", ["Annualized", "Total"], horizontal=True)

if mode == "Annualized":
    eff_df = ranked_df[["Scenario", "Annualized (Net)", "Annualized Tax Drag", "End Net Value"]].rename(
        columns={"Annualized (Net)": "Return", "Annualized Tax Drag": "Tax Drag", "End Net Value": "End Net"}
    )
    x_title = "Annualized Net Return"
    y_title = "Annualized Tax Drag"
else:
    eff_df = ranked_df[["Scenario", "Total Return (Net)", "Total Return Tax Drag", "End Net Value"]].rename(
        columns={"Total Return (Net)": "Return", "Total Return Tax Drag": "Tax Drag", "End Net Value": "End Net"}
    )
    x_title = "Total Net Return"
    y_title = "Total Tax Drag"

# Determine axis ranges dynamically with padding
x_min, x_max = float(eff_df["Return"].min()), float(eff_df["Return"].max())
y_min, y_max = float(eff_df["Tax Drag"].min()), float(eff_df["Tax Drag"].max())

x_pad = (x_max - x_min) * 0.10 if x_max != x_min else 0.01
y_pad = (y_max - y_min) * 0.10 if y_max != y_min else 0.01

x_domain = [x_min - x_pad, x_max + x_pad]
y_domain = [y_min - y_pad, y_max + y_pad]

scatter = (
    alt.Chart(eff_df)
    .mark_circle(opacity=0.85)
    .encode(
        x=alt.X("Return:Q", axis=alt.Axis(format=".1%", title=x_title), scale=alt.Scale(domain=x_domain, zero=False, nice=True)),
        y=alt.Y("Tax Drag:Q", axis=alt.Axis(format=".1%", title=y_title), scale=alt.Scale(domain=y_domain, zero=False, nice=True)),
        size=alt.Size("End Net:Q", title="End Net Value ($)", scale=alt.Scale(zero=False)),
        color=alt.Color("Scenario:N", legend=alt.Legend(title="Scenario")),
        tooltip=[
            alt.Tooltip("Scenario:N"),
            alt.Tooltip("Return:Q", format=".2%"),
            alt.Tooltip("Tax Drag:Q", format=".2%"),
            alt.Tooltip("End Net:Q", format=",.0f"),
        ],
    )
    .properties(height=420)
)

labels = (
    alt.Chart(eff_df)
    .mark_text(align="left", dx=8, dy=0)
    .encode(x="Return:Q", y="Tax Drag:Q", text="Scenario:N", color="Scenario:N")
)

st.altair_chart(scatter + labels, use_container_width=True)

# Footnotes & deployment tips
with st.expander("Modeling Notes & Assumptions"):
    st.markdown(
        """
- Version 1.5: Taxes & PPLI cost use number inputs; results ranked by Ending Net Value; Tax Efficiency chart now auto-scales axes.
- Global: expected returns, tax mix, tax rates, PPLI cost, and strategy. Per-scenario: allocations, PPLI sleeve, and names.
- All modeling is annual and simplified. This is educational, not tax or investment advice.
        """
    )

with st.expander("How to deploy on Streamlit Cloud / GitHub"):
    st.markdown(
        """
1. Create a GitHub repo with this file named `streamlit_app.py` at the root.  
2. Add a `requirements.txt` with:  
   ```
   streamlit
   pandas
   numpy
   altair
   ```
3. Push to GitHub. In Streamlit Community Cloud, choose **New app** and point to your repo & branch.  
4. Set **Main file path** to `streamlit_app.py`, then deploy.
        """
    )
