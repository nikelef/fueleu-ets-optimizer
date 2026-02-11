from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

from core.fuel_data import load_fuels_csv, fuels_as_dict
from core.etscalc import compute_ets, EtsInputs
from core.fueleucalc import compute_fueleu, FuelEUInputs
from core.optimize import optimize_policy, OptimizeInputs
from core.config import DefaultPrices, fueleu_target_intensity

st.set_page_config(page_title="FuelEU + EU ETS Optimizer (2026+)", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
FUELS_CSV = DATA_DIR / "fuels_default.csv"

st.title("FuelEU Maritime + EU ETS Cost Calculator & Optimizer (Reporting Years 2026+)")

# ---------------------------
# Load fuels
# ---------------------------
fuels_df = load_fuels_csv(FUELS_CSV)
fuels = fuels_as_dict(fuels_df)

with st.sidebar:
    st.header("General")
    year = st.number_input("Reporting year (>= 2026)", min_value=2026, max_value=2050, value=2026, step=1)

    st.header("Prices")
    defaults = DefaultPrices()
    eua_price = st.number_input("EUA price (EUR / tCO2e)", min_value=0.0, value=float(defaults.eua_price_eur_per_tco2e), step=1.0)
    pool_price = st.number_input("FuelEU pooling transfer price (EUR / tCO2e)", min_value=0.0, value=float(defaults.pool_price_eur_per_tco2e), step=1.0)

    st.header("Buckets (annual)")
    st.caption("Buckets map to EU scope factors: intra-EU=100%, extra-EU=50%, berth=100%.")

# ---------------------------
# Consumption input
# ---------------------------
st.subheader("1) Annual fuel consumption input (tonnes)")

fuel_keys = list(fuels.keys())
fuel_labels = {k: fuels[k]["fuel_name"] for k in fuel_keys}

colA, colB, colC = st.columns(3)

def bucket_editor(bucket_name: str) -> pd.DataFrame:
    df0 = pd.DataFrame({
        "fuel_key": fuel_keys,
        "fuel_name": [fuel_labels[k] for k in fuel_keys],
        "tonnes": [0.0 for _ in fuel_keys],
    })
    edited = st.data_editor(
        df0,
        key=f"editor_{bucket_name}",
        hide_index=True,
        column_config={
            "tonnes": st.column_config.NumberColumn("Tonnes (annual)", min_value=0.0, step=10.0),
            "fuel_key": st.column_config.TextColumn("fuel_key", disabled=True),
            "fuel_name": st.column_config.TextColumn("Fuel", disabled=True),
        },
        use_container_width=True,
    )
    return edited

with colA:
    st.markdown("**Intra-EU voyages**")
    df_intra = bucket_editor("intra_eu")
with colB:
    st.markdown("**Extra-EU voyages** (EU/EEA ↔ non-EU/EEA)")
    df_extra = bucket_editor("extra_eu")
with colC:
    st.markdown("**At berth (EU/EEA ports)**")
    df_berth = bucket_editor("berth")

def df_to_bucket(df: pd.DataFrame) -> dict:
    out = {}
    for _, r in df.iterrows():
        t = float(r["tonnes"])
        if t > 0:
            out[str(r["fuel_key"])] = t
    return out

consumption = {
    "intra_eu": df_to_bucket(df_intra),
    "extra_eu": df_to_bucket(df_extra),
    "berth": df_to_bucket(df_berth),
}

# ---------------------------
# Fuel prices input
# ---------------------------
st.subheader("2) Fuel prices (EUR/tonne)")

price_df0 = pd.DataFrame({
    "fuel_key": fuel_keys,
    "fuel_name": [fuel_labels[k] for k in fuel_keys],
    "eur_per_tonne": [650.0 if "MGO" in k else 600.0 for k in fuel_keys],
})
price_df = st.data_editor(
    price_df0,
    key="price_editor",
    hide_index=True,
    column_config={
        "eur_per_tonne": st.column_config.NumberColumn("EUR/tonne", min_value=0.0, step=10.0),
        "fuel_key": st.column_config.TextColumn("fuel_key", disabled=True),
        "fuel_name": st.column_config.TextColumn("Fuel", disabled=True),
    },
    use_container_width=True,
)

fuel_prices = {str(r["fuel_key"]): float(r["eur_per_tonne"]) for _, r in price_df.iterrows()}

# ---------------------------
# Compute ETS + FuelEU
# ---------------------------
st.subheader("3) Results (calculator mode)")

run_calc = st.button("Run calculation", type="primary")

if run_calc:
    ets = compute_ets(
        fuels,
        EtsInputs(
            reporting_year=int(year),
            eua_price_eur_per_tco2e=float(eua_price),
            consumption_tonnes=consumption,
        ),
    )

    # Wind policy in calculator mode
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        apply_wind = st.checkbox("Apply wind-assist reward factor (FuelEU)", value=False)
    with c2:
        wind_ratio = st.selectbox("Pwind/Pprop", options=[0.0, 0.05, 0.10, 0.15], index=0)
    fueleu = compute_fueleu(
        fuels,
        FuelEUInputs(
            reporting_year=int(year),
            consumption_tonnes=consumption,
            wind_pwind_over_pprop=float(wind_ratio),
            apply_wind_reward=bool(apply_wind and wind_ratio > 0),
        ),
    )

    # Summaries
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("EU ETS cost (EUR)", f"{ets['totals']['ets_cost_eur']:,.0f}")
    s2.metric("ETS surrender tCO2e", f"{ets['totals']['co2e_t_surrender']:,.0f}")
    s3.metric("FuelEU actual (gCO2eq/MJ)", f"{fueleu['actual_rewarded_gco2eq_per_mj']:.2f}")
    s4.metric("FuelEU penalty (EUR)", f"{fueleu['penalty_eur']:,.0f}")

    st.caption(f"FuelEU target for {int(year)}: {fueleu_target_intensity(int(year)):.2f} gCO2eq/MJ")

    # ETS breakdown chart
    bdf = pd.DataFrame(ets["breakdown"])
    if not bdf.empty:
        fig1 = px.bar(
            bdf,
            x="fuel_name",
            y="co2e_t_surrender",
            color="bucket",
            title="EU ETS: surrendered tCO2e by fuel and bucket",
        )
        st.plotly_chart(fig1, use_container_width=True)

    # FuelEU breakdown chart
    fdf = pd.DataFrame(fueleu["rows"])
    if not fdf.empty:
        fig2 = px.bar(
            fdf,
            x="fuel_name",
            y="energy_mj_scoped",
            color="bucket",
            title="FuelEU: scoped energy (MJ) by fuel and bucket",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Detailed tables**")
    t1, t2 = st.columns(2)
    with t1:
        st.write("EU ETS breakdown")
        st.dataframe(bdf, use_container_width=True)
    with t2:
        st.write("FuelEU breakdown")
        st.dataframe(fdf, use_container_width=True)

# ---------------------------
# Optimizer
# ---------------------------
st.subheader("4) Optimizer (policy selection)")

st.markdown(
    """
The optimizer chooses fuel quantities to meet **energy demand (MJ)** in each bucket, then evaluates:
- Fuel cost + EU ETS cost (2026+ includes CH₄ and N₂O in CO₂e)
- FuelEU penalty, or (optionally) **pooling** buy/sell at an internal transfer price
- Optional **wind-assist reward factor** (discrete options)
"""
)

opt_col1, opt_col2 = st.columns([1, 2])

with opt_col1:
    st.markdown("**Energy demand (MJ)**")
    dem_intra = st.number_input("Intra-EU demand (MJ)", min_value=0.0, value=1.0e9, step=1.0e8, format="%.0f")
    dem_extra = st.number_input("Extra-EU demand (MJ)", min_value=0.0, value=1.0e9, step=1.0e8, format="%.0f")
    dem_berth = st.number_input("Berth demand (MJ)", min_value=0.0, value=1.0e8, step=1.0e7, format="%.0f")

    allow_pooling = st.checkbox("Allow FuelEU pooling (buy/sell compliance balance)", value=True)
    allow_wind = st.checkbox("Allow wind-assist reward factor choice", value=True)

with opt_col2:
    st.markdown("**Constraints (max energy share per bucket)**")
    st.caption("Set max share for a few fuels; everything else is effectively unlimited unless you set it.")
    max_df0 = pd.DataFrame({
        "bucket": ["intra_eu", "extra_eu", "berth"],
        "fuel_key": ["MGO", "LNG_DF_MS", "MGO"],
        "max_share": [1.0, 0.7, 1.0],
    })
    max_df = st.data_editor(
        max_df0,
        key="maxshare_editor",
        hide_index=True,
        column_config={
            "bucket": st.column_config.SelectboxColumn("Bucket", options=["intra_eu", "extra_eu", "berth"]),
            "fuel_key": st.column_config.SelectboxColumn("Fuel", options=fuel_keys),
            "max_share": st.column_config.NumberColumn("Max share (0..1)", min_value=0.0, max_value=1.0, step=0.05),
        },
        use_container_width=True,
    )

def build_max_share(dfx: pd.DataFrame) -> dict:
    out = {}
    for _, r in dfx.iterrows():
        b = str(r["bucket"])
        fk = str(r["fuel_key"])
        mx = float(r["max_share"])
        out.setdefault(b, {})[fk] = mx
    return out

do_opt = st.button("Run optimizer", type="primary", key="run_optimizer")

if do_opt:
    max_share = build_max_share(max_df)

    opt_inp = OptimizeInputs(
        reporting_year=int(year),
        energy_demand_mj={
            "intra_eu": float(dem_intra),
            "extra_eu": float(dem_extra),
            "berth": float(dem_berth),
        },
        max_share=max_share,
        fuel_price_eur_per_tonne=fuel_prices,
        eua_price_eur_per_tco2e=float(eua_price),
        pool_price_eur_per_tco2e=float(pool_price),
        allow_pooling=bool(allow_pooling),
        allow_wind_reward=bool(allow_wind),
        wind_options=[0.0, 0.05, 0.10, 0.15],
    )

    res = optimize_policy(fuels, opt_inp)

    st.success("Optimisation complete.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total cost (EUR)", f"{res['total_cost_eur']:,.0f}")
    m2.metric("Fuel cost (EUR)", f"{res['fuel_cost_eur']:,.0f}")
    m3.metric("EU ETS cost (EUR)", f"{res['ets']['totals']['ets_cost_eur']:,.0f}")
    m4.metric("FuelEU pooling trade (EUR)", f"{res['fueleu_pool_trade_eur']:,.0f}")

    st.write(f"Chosen wind option Pwind/Pprop: **{res['chosen_wind_pwind_over_pprop']:.2f}**")

    sol_df = []
    for bucket, by_fuel in res["solution_tonnes"].items():
        for fk, t in by_fuel.items():
            sol_df.append({
                "bucket": bucket,
                "fuel_key": fk,
                "fuel_name": fuels[fk]["fuel_name"],
                "tonnes": t,
                "eur_per_tonne": fuel_prices.get(fk, 0.0),
                "cost_eur": t * fuel_prices.get(fk, 0.0),
            })
    sol_df = pd.DataFrame(sol_df).sort_values(["bucket", "cost_eur"], ascending=[True, False])
    st.subheader("Optimized fuel plan (tonnes)")
    st.dataframe(sol_df, use_container_width=True)

    if not sol_df.empty:
        fig = px.bar(sol_df, x="fuel_name", y="tonnes", color="bucket", title="Optimized tonnes by fuel and bucket")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("FuelEU outcome (optimized)")
    st.json({
        "target_gco2eq_per_mj": res["fueleu"]["target_gco2eq_per_mj"],
        "actual_rewarded_gco2eq_per_mj": res["fueleu"]["actual_rewarded_gco2eq_per_mj"],
        "compliance_balance_gco2eq": res["fueleu"]["compliance_balance_gco2eq"],
        "penalty_eur (if not pooling)": res["fueleu"]["penalty_eur"],
    })
