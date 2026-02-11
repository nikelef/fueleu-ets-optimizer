from __future__ import annotations

import pandas as pd
from typing import Dict

def tonnes_dict_to_df(consumption_tonnes: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for bucket, by_fuel in consumption_tonnes.items():
        for fk, t in by_fuel.items():
            rows.append({"bucket": bucket, "fuel_key": fk, "tonnes": t})
    return pd.DataFrame(rows)
