from __future__ import annotations

import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

@dataclass
class FuelRow:
    fuel_key: str
    fuel_name: str
    lcv_mj_per_g: float
    wtt_gco2eq_per_mj: float
    cf_co2_g_per_gfuel: float
    cf_ch4_g_per_gfuel: float
    cf_n2o_g_per_gfuel: float
    slip_pct: float

def load_fuels_csv(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Clean up blanks
    for c in ["cf_ch4_g_per_gfuel", "cf_n2o_g_per_gfuel", "slip_pct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

def fuels_as_dict(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for _, r in df.iterrows():
        out[str(r["fuel_key"])] = {
            "fuel_key": str(r["fuel_key"]),
            "fuel_name": str(r["fuel_name"]),
            "lcv_mj_per_g": float(r["lcv_mj_per_g"]),
            "wtt_gco2eq_per_mj": float(r["wtt_gco2eq_per_mj"]),
            "cf_co2_g_per_gfuel": float(r["cf_co2_g_per_gfuel"]),
            "cf_ch4_g_per_gfuel": float(r.get("cf_ch4_g_per_gfuel", 0.0)),
            "cf_n2o_g_per_gfuel": float(r.get("cf_n2o_g_per_gfuel", 0.0)),
            "slip_pct": float(r.get("slip_pct", 0.0)),
            "notes": str(r.get("notes", "")),
        }
    return out
