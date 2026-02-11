from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import pulp

from .etscalc import compute_ets, EtsInputs
from .fueleucalc import compute_fueleu, FuelEUInputs
from .config import DefaultPrices

@dataclass
class OptimizeInputs:
    reporting_year: int

    # Total fuel demand per bucket (tonnes of "base fuel equivalent" is not valid; we keep tonnes by fuel decision)
    # Instead: you provide a total ENERGY demand (MJ) per bucket that must be met by chosen fuels.
    energy_demand_mj: Dict[str, float]  # bucket -> MJ

    # Allowed fuels and their max shares per bucket (0..1); if missing => allowed
    max_share: Dict[str, Dict[str, float]]  # bucket -> fuel_key -> max_share

    # Fuel prices in EUR/tonne by fuel_key
    fuel_price_eur_per_tonne: Dict[str, float]

    # ETS / pooling price inputs
    eua_price_eur_per_tco2e: float
    pool_price_eur_per_tco2e: float

    # Policies toggles
    allow_pooling: bool = True
    allow_wind_reward: bool = True

    # Wind selection (decision) as discrete options of Pwind/Pprop
    wind_options: List[float] = None  # e.g. [0.0, 0.05, 0.10, 0.15]

def optimize_policy(fuels: Dict[str, Dict[str, Any]], inp: OptimizeInputs) -> Dict[str, Any]:
    """
    Mixed strategy optimizer:
    - chooses fuel tonnes per bucket to satisfy bucket energy demand
    - optionally chooses wind-assist option (discrete)
    - optionally uses pooling by buying/selling FuelEU compliance at a user-defined price (EUR/tCO2e)
      (Implemented as "make deficit zero by purchasing CB" or monetise surplus.)
    Objective:
    - Minimise: fuel_cost + ETS_cost + FuelEU_net_cost (penalty or pooling trade)
    """
    if inp.wind_options is None:
        inp.wind_options = [0.0, 0.05, 0.10, 0.15]

    buckets = list(inp.energy_demand_mj.keys())
    fuel_keys = list(fuels.keys())

    prob = pulp.LpProblem("FuelEU_ETS_Optimization", pulp.LpMinimize)

    # Decision: tonnes of each fuel used in each bucket
    x = pulp.LpVariable.dicts("tonnes", (buckets, fuel_keys), lowBound=0)

    # Discrete wind choice (one-hot)
    y = pulp.LpVariable.dicts("wind_choice", list(range(len(inp.wind_options))), lowBound=0, upBound=1, cat="Binary")
    prob += pulp.lpSum([y[i] for i in y]) == 1.0

    # Energy constraints per bucket: sum(tonnes * 1e6 g/t * LCV MJ/g) >= demand
    for b in buckets:
        prob += pulp.lpSum([
            x[b][fk] * 1_000_000.0 * fuels[fk]["lcv_mj_per_g"]
            for fk in fuel_keys
        ]) >= float(inp.energy_demand_mj[b]), f"energy_{b}"

        # Max share constraints (by energy) if provided
        if b in inp.max_share:
            demand = float(inp.energy_demand_mj[b])
            for fk, mx in inp.max_share[b].items():
                mx = float(mx)
                prob += (x[b][fk] * 1_000_000.0 * fuels[fk]["lcv_mj_per_g"]) <= (mx * demand), f"maxshare_{b}_{fk}"

    # Fuel cost
    fuel_cost = pulp.lpSum([
        x[b][fk] * float(inp.fuel_price_eur_per_tonne.get(fk, 0.0))
        for b in buckets for fk in fuel_keys
    ])

    # Build a helper to evaluate ETS and FuelEU from decision vars by introducing auxiliary variables is complex.
    # So we solve in two stages:
    # 1) Optimise fuel cost + linearised ETS proxy (CO2e per tonne approximation)
    # 2) Recompute exact ETS + FuelEU and do a local search over wind options & pooling.
    #
    # Stage-1 uses a conservative CO2e proxy per tonne = (Cf_CO2 + 28*Cf_CH4 + 265*Cf_N2O + slip%*28)*1e6 g -> tonnes CO2e
    co2e_proxy_per_tonne = {}
    for fk in fuel_keys:
        f = fuels[fk]
        g_per_g = float(f["cf_co2_g_per_gfuel"]) \
                  + 28.0 * float(f.get("cf_ch4_g_per_gfuel", 0.0)) \
                  + 265.0 * float(f.get("cf_n2o_g_per_gfuel", 0.0))
        slip = float(f.get("slip_pct", 0.0))
        if slip > 0:
            g_per_g += 28.0 * (slip / 100.0)  # treat slip as CH4
        # tonnes fuel -> grams fuel -> grams CO2e -> tonnes CO2e
        co2e_proxy_per_tonne[fk] = (1_000_000.0 * g_per_g) / 1_000_000.0  # simplifies to g_per_g (tonnes CO2e per tonne fuel)

    # ETS proxy with scope factors per bucket (same mapping)
    bucket_scope = {"intra_eu": 1.0, "extra_eu": 0.5, "berth": 1.0}

    ets_proxy_tco2e = pulp.lpSum([
        x[b][fk] * co2e_proxy_per_tonne[fk] * bucket_scope[b]
        for b in buckets for fk in fuel_keys
    ])

    ets_proxy_cost = ets_proxy_tco2e * float(inp.eua_price_eur_per_tco2e)  # surrender=1 for 2026+ user case

    prob += fuel_cost + ets_proxy_cost

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Optimisation failed: {pulp.LpStatus[status]}")

    # Extract solution tonnes
    sol_tonnes = {b: {} for b in buckets}
    for b in buckets:
        for fk in fuel_keys:
            v = float(x[b][fk].value())
            if v > 1e-9:
                sol_tonnes[b][fk] = v

    wind_idx = max(y.keys(), key=lambda i: float(y[i].value()))
    chosen_wind = inp.wind_options[int(wind_idx)]

    # Recompute exact ETS + FuelEU with the chosen wind option
    ets = compute_ets(
        fuels,
        EtsInputs(
            reporting_year=inp.reporting_year,
            eua_price_eur_per_tco2e=inp.eua_price_eur_per_tco2e,
            consumption_tonnes=sol_tonnes,
        ),
    )

    fueleu = compute_fueleu(
        fuels,
        FuelEUInputs(
            reporting_year=inp.reporting_year,
            consumption_tonnes=sol_tonnes,
            wind_pwind_over_pprop=chosen_wind,
            apply_wind_reward=bool(inp.allow_wind_reward and chosen_wind > 0),
        ),
    )

    # Pooling model (simple economic layer):
    # - If deficit (CB < 0): buy CB to bring it to 0 at pool_price per tCO2e
    # - If surplus (CB > 0): sell CB at pool_price per tCO2e (user-defined)
    pool_trade_eur = 0.0
    cb_t = float(fueleu["compliance_balance_gco2eq"]) / 1e6  # g -> tonnes
    if inp.allow_pooling:
        pool_trade_eur = (-cb_t) * float(inp.pool_price_eur_per_tco2e)  # deficit => positive cost

        # If surplus => negative cost (revenue)
        # But never allow “double counting” with penalties:
        # If we pool, we assume you avoid penalty by bringing CB to >= 0.
        if cb_t < 0:
            fueleu_penalty_effective = 0.0
        else:
            fueleu_penalty_effective = 0.0
    else:
        pool_trade_eur = 0.0
        fueleu_penalty_effective = float(fueleu["penalty_eur"])

    total_fuel_cost = sum(
        sol_tonnes[b].get(fk, 0.0) * float(inp.fuel_price_eur_per_tonne.get(fk, 0.0))
        for b in buckets for fk in sol_tonnes[b].keys()
    )

    total_cost = total_fuel_cost + float(ets["totals"]["ets_cost_eur"]) + fueleu_penalty_effective + pool_trade_eur

    return {
        "solution_tonnes": sol_tonnes,
        "chosen_wind_pwind_over_pprop": chosen_wind,
        "fuel_cost_eur": total_fuel_cost,
        "ets": ets,
        "fueleu": fueleu,
        "fueleu_pool_trade_eur": pool_trade_eur,
        "fueleu_penalty_effective_eur": fueleu_penalty_effective,
        "total_cost_eur": total_cost,
    }
