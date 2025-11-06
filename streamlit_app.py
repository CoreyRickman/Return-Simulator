import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

st.set_page_config(page_title="PPLI vs Taxable Return Model", layout="wide")

# -----------------------------
# Helpers
# -----------------------------

ASSETS = ["Stocks", "Bonds", "Alternatives"]
STRATEGIES = {
    "Buy & Hold (no annual rebalance)": "buy_hold",
    "Rebalanced annually to target weights": "rebalanced",
}

@st.cache_data
def _years_list(T: int) -> List[int]:
    return list(range(0, T + 1))


def normalize_percents(values: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(v, 0.0) for v in values.values())
    if total == 0:
        # default 100% to first key if all zeros
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
    asset_returns_pct: Dict[str, float],
    ppli_cost_pct: float,
    sleeve_pct_in_ppli: Dict[str, float],  # 0..100 per asset
    tax_ordinary_pct: float,
    tax_cg_pct: float,
    tax_mix_pct: Dict[str, Tuple[float, float]],  # per asset: (pct ordinary, pct cap gains) sums ~100
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    # Taxable net: r * (1 - (mix_OI*taxOI + mix_CG*taxCG))  [applied annually]
    r_taxable_net = {}
    r_ppli_net = {}
    for a in ASSETS:
        mix_oi = pct_to_decimal(tax_mix[a]["OI"])  # in decimals summing to 1
        mix_cg = pct_to_decimal(tax_mix[a]["CG"])  # in decimals summing to 1
        blend_tax = mix_oi * tax_oi + mix_cg * tax_cg
        r_taxable_net[a] = r_gross[a] * (1.0 - blend_tax)
        # PPLI modeled as a simple annual cost drag
        r_ppli_net[a] = r_gross[a] - ppli_cost

    years_list = _years_list(years)

    # Initialize holdings for gross and net
    # For net, we track taxable and ppli sleeves separately per asset
    gross_values = {a: [initial * pct_to_decimal(w[a])] for a in ASSETS}
    net_taxable_values = {a: [initial * pct_to_decimal(w[a]) * (1.0 - pct_to_decimal(sleeve_ppli[a]))] for a in ASSETS}
    net_ppli_values = {a: [initial * pct_to_decimal(w[a]) * pct_to_decimal(sleeve_ppli[a])] for a in ASSETS}

    # Simulation loop
    for t in years_list[1:]:
        # Rebalance at the start of the year if strategy is rebalanced
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
            # Gross growth (ignoring taxes/costs)
            gross_values[a].append(gross_values[a][-1] * (1.0 + r_gross[a]))
            # Net growth for taxable and ppli sleeves
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

st.title("📈 PPLI vs Taxable: Multi‑Scenario Return Model (v1)")

with st.sidebar:
    st.header("Global Settings")
    initial = st.number_input("Initial Investment ($)", min_value=1000.0, value=1_000_000.0, step=1000.0, format="%0.2f")
    years = st.number_input("Investment Horizon (years)", min_value=1, max_value=100, value=20, step=1)

    st.subheader("Tax Assumptions (% per year)")
    tax_ordinary_pct = st.slider("Ordinary Income Tax %", min_value=0.0, max_value=70.0, value=37.0, step=0.1)
    tax_cg_pct = st.slider("Capital Gains Tax %", min_value=0.0, max_value=70.0, value=23.8, step=0.1)

    st.subheader("PPLI")
    ppli_cost_pct = st.slider("PPLI Annual Cost %", min_value=0.0, max_value=5.0, value=1.0, step=0.05)

    st.caption("Notes: All modeling is annual and simplified. Taxes are applied as annual drags to the taxable sleeve; PPLI is modeled as an annual cost drag only.")

# -----------------------------
# Scenario builder
# -----------------------------

st.markdown("## Scenarios (choose allocations & assumptions for each)")

scenario_tabs = st.tabs([f"Scenario {i+1}" for i in range(4)])

scenario_inputs = []

for i, tab in enumerate(scenario_tabs):
    with tab:
        st.subheader(f"Scenario {i+1} Inputs")
        cols = st.columns(3)

        # Portfolio weights
        st.markdown("**Portfolio Allocation (%)** — must total ~100%")
        w1, w2, w3 = st.columns(3)
        w_stocks = w1.number_input(f"Stocks % (S{i+1})", min_value=0.0, max_value=100.0, value=60.0, step=1.0, key=f"w_stocks_{i}")
        w_bonds = w2.number_input(f"Bonds % (S{i+1})", min_value=0.0, max_value=100.0, value=30.0, step=1.0, key=f"w_bonds_{i}")
        w_alts = w3.number_input(f"Alternatives % (S{i+1})", min_value=0.0, max_value=100.0, value=10.0, step=1.0, key=f"w_alts_{i}")
        weights = {"Stocks": w_stocks, "Bonds": w_bonds, "Alternatives": w_alts}
        w_sum = sum(weights.values())
        if abs(w_sum - 100.0) > 0.01:
            st.warning(f"Allocations sum to {w_sum:.2f}%. They will be normalized to 100% for calculations.")

        st.markdown("**Expected Annual Returns (% - before tax & PPLI)**")
        r1, r2, r3 = st.columns(3)
        r_stocks = r1.number_input(f"Stocks r% (S{i+1})", min_value=-50.0, max_value=50.0, value=7.0, step=0.1, key=f"r_stocks_{i}")
        r_bonds = r2.number_input(f"Bonds r% (S{i+1})", min_value=-50.0, max_value=50.0, value=4.0, step=0.1, key=f"r_bonds_{i}")
        r_alts = r3.number_input(f"Alternatives r% (S{i+1})", min_value=-50.0, max_value=50.0, value=6.0, step=0.1, key=f"r_alts_{i}")
        returns = {"Stocks": r_stocks, "Bonds": r_bonds, "Alternatives": r_alts}

        st.markdown("**Tax Mix of Returns (%, per asset)** — portion taxed as OI vs CG (normalized to 100%)")
        tm1, tm2, tm3 = st.columns(3)
        st_oi = tm1.number_input(f"Stocks % Ordinary (S{i+1})", min_value=0.0, max_value=100.0, value=20.0, step=1.0, key=f"tm_st_oi_{i}")
        st_cg = tm1.number_input(f"Stocks % Cap Gains (S{i+1})", min_value=0.0, max_value=100.0, value=80.0, step=1.0, key=f"tm_st_cg_{i}")
        bo_oi = tm2.number_input(f"Bonds % Ordinary (S{i+1})", min_value=0.0, max_value=100.0, value=100.0, step=1.0, key=f"tm_bo_oi_{i}")
        bo_cg = tm2.number_input(f"Bonds % Cap Gains (S{i+1})", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key=f"tm_bo_cg_{i}")
        al_oi = tm3.number_input(f"Alts % Ordinary (S{i+1})", min_value=0.0, max_value=100.0, value=50.0, step=1.0, key=f"tm_al_oi_{i}")
        al_cg = tm3.number_input(f"Alts % Cap Gains (S{i+1})", min_value=0.0, max_value=100.0, value=50.0, step=1.0, key=f"tm_al_cg_{i}")
        tax_mix = {
            "Stocks": (st_oi, st_cg),
            "Bonds": (bo_oi, bo_cg),
            "Alternatives": (al_oi, al_cg),
        }

        st.markdown("**PPLI Sleeve (% of each asset held in PPLI)**")
        p1, p2, p3 = st.columns(3)
        p_st = p1.number_input(f"Stocks in PPLI % (S{i+1})", min_value=0.0, max_value=100.0, value=50.0, step=1.0, key=f"p_st_{i}")
        p_bo = p2.number_input(f"Bonds in PPLI % (S{i+1})", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key=f"p_bo_{i}")
        p_al = p3.number_input(f"Alts in PPLI % (S{i+1})", min_value=0.0, max_value=100.0, value=50.0, step=1.0, key=f"p_al_{i}")
        sleeve = {"Stocks": p_st, "Bonds": p_bo, "Alternatives": p_al}

        strategy_label = st.selectbox(
            "Strategy",
            list(STRATEGIES.keys()),
            index=1,
            key=f"strat_{i}",
        )

        scenario_inputs.append({
            "weights": weights,
            "returns": returns,
            "tax_mix": tax_mix,
            "sleeve": sleeve,
            "strategy": STRATEGIES[strategy_label],
        })

# -----------------------------
# Run simulations
# -----------------------------

results: List[Dict] = []
series_net: Dict[str, pd.Series] = {}
series_gross: Dict[str, pd.Series] = {}

for idx, sc in enumerate(scenario_inputs):
    gross_df, net_df = simulate_scenario(
        initial=initial,
        years=years,
        strategy=sc["strategy"],
        asset_weights_pct=sc["weights"],
        asset_returns_pct=sc["returns"],
        ppli_cost_pct=ppli_cost_pct,
        sleeve_pct_in_ppli=sc["sleeve"],
        tax_ordinary_pct=tax_ordinary_pct,
        tax_cg_pct=tax_cg_pct,
        tax_mix_pct=sc["tax_mix"],
    )
    metrics = summarize_metrics(initial, gross_df, net_df)
    results.append({"Scenario": f"Scenario {idx+1}", **metrics})
    series_net[f"Scenario {idx+1} (Net)"] = net_df["Total"]
    series_gross[f"Scenario {idx+1} (Gross)"] = gross_df["Total"]

# -----------------------------
# Outputs: Metrics + Chart
# -----------------------------

st.markdown("## Results")

metrics_df = pd.DataFrame(results)
metrics_df_display = metrics_df.copy()

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

st.markdown("### Growth Over Time")
show_gross = st.checkbox("Show Gross lines (untaxed, no PPLI cost)", value=False)

plot_df = pd.DataFrame(index=_years_list(years))
for name, s in series_net.items():
    plot_df[name] = s
if show_gross:
    for name, s in series_gross.items():
        plot_df[name] = s

plot_df.index.name = "Year"

# Streamlit's built-in line chart (Altair under the hood)
st.line_chart(plot_df, height=420)

# -----------------------------
# Footnotes & assumptions
# -----------------------------
with st.expander("Modeling Notes & Assumptions"):
    st.markdown(
        """
- All returns, taxes, and costs are applied annually (year-end).  
- **Gross** results ignore all taxes and PPLI costs.  
- **Net** results apply blended annual tax drag to the taxable sleeve only and a flat PPLI cost drag to the PPLI sleeve only.  
- The *Tax Mix of Returns* (Ordinary vs. Cap Gains) is normalized to 100% per asset.  
- Negative returns are treated symmetrically for simplicity (i.e., the same blended tax rate is applied to the annual return, which may understate tax advantages of losses); this is a Version 1 simplification.  
- *Buy & Hold* tracks asset and sleeve values without rebalancing; *Rebalanced* resets to target weights and sleeves at the start of each year.  
- This tool is educational and illustrative; not tax or investment advice.
        """
    )

# -----------------------------
# Deployment tips
# -----------------------------
with st.expander("How to deploy on Streamlit Cloud / GitHub"):
    st.markdown(
        """
1. Create a GitHub repo with this file named `streamlit_app.py` at the root.  
2. Add a `requirements.txt` with:  
   ```
   streamlit
   pandas
   numpy
   ```
3. Push to GitHub. In [Streamlit Community Cloud](https://share.streamlit.io), choose **New app** and point to your repo & branch.  
4. Set **Main file path** to `streamlit_app.py`, then deploy.
        """
    )
