# FuelEU Maritime + EU ETS Optimizer (2026+)

A Streamlit app that:
- Calculates EU ETS (maritime) emissions (CO2 + CH4 + N2O) and EUA cost
- Calculates FuelEU Maritime WtW GHG intensity, compliance balance, and penalty
- Optimizes fuel choices under policies:
  - pooling (buy/sell compliance balance at an internal transfer price),
  - alternative fuel/blending,
  - wind-assist reward factor (discrete options).

## Run locally

```bash
python -m venv .venv
# activate venv
pip install -r requirements.txt
streamlit run app.py
