# Market Dashboard (Streamlit)

This is a self-contained Python dashboard that shows:
- VIX (level + regimes + rolling correlation vs SPY)
- Fear & Greed (best-effort fetch; app still works if the endpoint fails)
- Sector rotation (RRG-style proxies: RS Ratio + RS Momentum)
- Sector performance heatmap

## Run
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## What “Sector Rotation” means here
- **RS Ratio** = z-scored level of (Sector price / SPY)
- **RS Momentum** = z-scored change of that ratio over your chosen momentum window
- Quadrants:
  - Leading: RS Ratio >= 0 and RS Momentum >= 0
  - Weakening: RS Ratio >= 0 and RS Momentum < 0
  - Lagging: RS Ratio < 0 and RS Momentum < 0
  - Improving: RS Ratio < 0 and RS Momentum >= 0

## Tips
- Use 1–3 years of data for cleaner rotation signals.
- Treat the RRG proxies as a **relative** positioning tool, not a prediction.
# financial-dashboard
