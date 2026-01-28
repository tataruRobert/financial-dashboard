"""
Market Dashboard: VIX + Fear & Greed + Sector Rotation

Run locally:
  1) python -m venv .venv
  2) source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
  3) pip install -r requirements.txt
  4) streamlit run app.py

Notes:
- VIX + ETFs are fetched via yfinance (free, delayed).
- Fear & Greed: tries CNN's public endpoint; if it fails, the app will still run and show "Unavailable".
"""

import math
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Settings
# -----------------------------
st.set_page_config(page_title="Market Dashboard", layout="wide")

SECTOR_ETFS: Dict[str, str] = {
    "Communication Services (XLC)": "XLC",
    "Consumer Discretionary (XLY)": "XLY",
    "Consumer Staples (XLP)": "XLP",
    "Energy (XLE)": "XLE",
    "Financials (XLF)": "XLF",
    "Health Care (XLV)": "XLV",
    "Industrials (XLI)": "XLI",
    "Materials (XLB)": "XLB",
    "Real Estate (XLRE)": "XLRE",
    "Technology (XLK)": "XLK",
    "Utilities (XLU)": "XLU",
}
BENCHMARK = "SPY"
VIX_TICKER = "^VIX"

DEFAULT_START_DAYS = 365 * 3

# Helpful lookup: ticker -> human-readable sector name
TICKER_TO_LABEL = {v: k for k, v in SECTOR_ETFS.items()}


# -----------------------------
# Utilities
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60 * 15)
def yf_history(tickers, start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Fetch adjusted close for one or many tickers.
    Returns a DataFrame indexed by date, columns=tickers.
    """
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    # yfinance returns different shapes for single vs multiple tickers
    if isinstance(tickers, str) or (isinstance(tickers, list) and len(tickers) == 1):
        # single
        adj = data["Close"].to_frame(name=tickers if isinstance(tickers, str) else tickers[0])
    else:
        # multi: columns like ('AAPL','Close')
        closes = []
        for t in tickers:
            if (t, "Close") in data.columns:
                closes.append(data[(t, "Close")].rename(t))
            elif "Close" in data.columns and t in data.columns.get_level_values(0):
                closes.append(data[t]["Close"].rename(t))
        adj = pd.concat(closes, axis=1)

    adj.index = pd.to_datetime(adj.index)
    adj = adj.dropna(how="all")
    return adj


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None


# -----------------------------
# Fear & Greed
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_fear_greed() -> Tuple[Optional[float], Optional[str], Optional[pd.DataFrame]]:
    """
    Attempts to fetch CNN Fear & Greed Index (0-100).
    Returns: (value, rating, historical_df)

    CNN sometimes changes endpoints; this is best-effort.
    """
    endpoints = [
        "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",  # common public endpoint
        "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/",
    ]
    last_err = None
    for url in endpoints:
        try:
            headers = {
                # CNN blocks obvious bots; mimic a normal browser.
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
                "Accept": "application/json,text/plain,*/*",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.cnn.com/",
                "Origin": "https://www.cnn.com",
            }
            r = requests.get(url, headers=headers, timeout=10)

            r.raise_for_status()
            js = r.json()
            # CNN has changed the payload over time:
            # - Older: fear_and_greed: { now: <float>, rating: <str>, historical: [...] }
            # - Newer (Jan 2026): fear_and_greed: { score: <float>, rating: <str>, ... }
            fg_block = js.get("fear_and_greed", {}) or js.get("fearAndGreed", {})
            raw_score = fg_block.get("now", fg_block.get("score"))
            rating = fg_block.get("rating")
            score = safe_float(raw_score)

            hist = None
            candidates = []

            # Preferred: dedicated historical block (current format)
            hist_block = js.get("fear_and_greed_historical")
            if isinstance(hist_block, dict):
                candidates = hist_block.get("data", [])

            # Fallback: older inline historical lists
            if not candidates:
                for key in ["fear_and_greed", "fearAndGreed", "data"]:
                    if key in js and isinstance(js[key], dict):
                        for hk in ["historical", "history", "data"]:
                            if hk in js[key] and isinstance(js[key][hk], list):
                                candidates = js[key][hk]
                                break
                    if candidates:
                        break

            if candidates:
                rows = []
                for item in candidates:
                    # Expect {"x": <ms since epoch>, "y": <score>}
                    if not isinstance(item, dict):
                        continue
                    x = item.get("x")
                    y = item.get("y")
                    if x is None or y is None:
                        continue
                    rows.append((pd.to_datetime(x, unit="ms"), safe_float(y)))
                if rows:
                    hist = pd.DataFrame(rows, columns=["date", "fear_greed"]).dropna()
                    hist = hist.set_index("date").sort_index()

            return score, rating, hist
        except Exception as e:
            last_err = e
            continue
    return None, None, None


# -----------------------------
# Sector Rotation / RRG-like
# -----------------------------
@dataclass
class RotationConfig:
    rs_window: int = 60          # window for RS ratio z-score
    mom_window: int = 20         # window for RS momentum (change in RS)
    z_window: int = 252          # window to standardize RS ratio


def compute_sector_rotation(prices: pd.DataFrame, benchmark: str, cfg: RotationConfig) -> pd.DataFrame:
    """
    Compute:
      - RS_Ratio: standardized relative strength ratio (sector / benchmark)
      - RS_Momentum: standardized momentum of RS ratio (delta over mom_window)
    """
    # Relative price ratio
    ratios = prices.div(prices[benchmark], axis=0)

    # Standardize ratio level (RRG-like RS-Ratio proxy)
    rs_ratio = ratios.apply(lambda s: rolling_zscore(s, cfg.z_window), axis=0)

    # Momentum = change in ratio over mom_window, then z-score
    mom_raw = ratios.pct_change(cfg.mom_window)
    rs_mom = mom_raw.apply(lambda s: rolling_zscore(s, cfg.z_window), axis=0)

    out = []
    last_date = prices.index.max()
    for col in prices.columns:
        if col == benchmark:
            continue
        out.append({
            "date": last_date,
            "ticker": col,
            "RS_Ratio": safe_float(rs_ratio[col].iloc[-1]),
            "RS_Momentum": safe_float(rs_mom[col].iloc[-1]),
            "Price_1M": safe_float(prices[col].pct_change(21).iloc[-1] * 100),
            "Price_3M": safe_float(prices[col].pct_change(63).iloc[-1] * 100),
            "Price_6M": safe_float(prices[col].pct_change(126).iloc[-1] * 100),
        })
    df = pd.DataFrame(out)
    return df


def quadrant(rs_ratio: float, rs_mom: float) -> str:
    if rs_ratio is None or rs_mom is None:
        return "Unknown"
    if rs_ratio >= 0 and rs_mom >= 0:
        return "Leading"
    if rs_ratio >= 0 and rs_mom < 0:
        return "Weakening"
    if rs_ratio < 0 and rs_mom < 0:
        return "Lagging"
    return "Improving"


# -----------------------------
# UI
# -----------------------------
st.title("📊 Market Dashboard — VIX, Fear & Greed, Sector Rotation")

with st.sidebar:
    st.header("Controls")
    today = dt.date.today()
    start = st.date_input("Start date", value=today - dt.timedelta(days=DEFAULT_START_DAYS))
    end = st.date_input("End date", value=today + dt.timedelta(days=1))
    st.caption("Tip: Use at least 1–3 years for cleaner rotation signals.")
    st.divider()

    st.subheader("Rotation settings")
    rs_z_window = st.slider("Standardization window (trading days)", 63, 504, 252, 21)
    mom_window = st.slider("Momentum lookback (days)", 5, 63, 20, 1)
    st.caption("RS_Ratio & RS_Momentum are z-scored proxies (RRG-style).")
    st.divider()

    st.subheader("Tickers")
    st.write(f"Benchmark: **{BENCHMARK}**")
    st.write(f"VIX: **{VIX_TICKER}**")


# Fetch data
tickers = [BENCHMARK, VIX_TICKER] + list(SECTOR_ETFS.values())
prices = yf_history(tickers, start=start, end=end)

# Ensure benchmark exists
if BENCHMARK not in prices.columns:
    st.error(f"Benchmark {BENCHMARK} not available from data source.")
    st.stop()

# VIX might have gaps; keep it separate
vix = prices[[VIX_TICKER]].dropna() if VIX_TICKER in prices.columns else pd.DataFrame()
sector_cols = [BENCHMARK] + list(SECTOR_ETFS.values())
sector_prices = prices[[c for c in sector_cols if c in prices.columns]].dropna(how="all")

# Fear & Greed
fg_score, fg_rating, fg_hist = fetch_fear_greed()

# -----------------------------
# Top KPI row
# -----------------------------
k1, k2, k3, k4 = st.columns(4)

# VIX KPI
if not vix.empty:
    vix_last = float(vix[VIX_TICKER].iloc[-1])
    vix_chg = float(vix[VIX_TICKER].pct_change().iloc[-1] * 100)
    k1.metric("VIX", f"{vix_last:,.2f}", f"{vix_chg:+.2f}%")
else:
    k1.metric("VIX", "Unavailable", "")

# Fear & Greed KPI
if fg_score is not None:
    k2.metric("Fear & Greed", f"{fg_score:.0f}/100", fg_rating or "")
else:
    k2.metric("Fear & Greed", "Unavailable", "")

# SPY KPI
spy_last = float(sector_prices[BENCHMARK].iloc[-1])
spy_chg = float(sector_prices[BENCHMARK].pct_change().iloc[-1] * 100)
k3.metric(f"{BENCHMARK}", f"{spy_last:,.2f}", f"{spy_chg:+.2f}%")

# Best/Worst sector 1M
one_month = sector_prices.pct_change(21).iloc[-1].dropna()
one_month = one_month.drop(index=[BENCHMARK], errors="ignore")
if len(one_month) > 0:
    best = one_month.idxmax()
    worst = one_month.idxmin()
    best_label = TICKER_TO_LABEL.get(best, best)
    worst_label = TICKER_TO_LABEL.get(worst, worst)
    # Two stacked metrics avoid truncation of long sector names inside a single metric box.
    with k4:
        st.metric("1M Leader", best_label, f"{one_month[best]*100:+.1f}%")
        st.metric("1M Laggard", worst_label, f"{one_month[worst]*100:+.1f}%")
else:
    k4.metric("1M Leaders / Laggards", "Unavailable", "")

st.divider()

tabs = st.tabs(["Overview", "VIX", "Fear & Greed", "Sector Rotation"])

# -----------------------------
# Overview
# -----------------------------
with tabs[0]:
    left, right = st.columns([2, 1])

    with left:
        st.subheader("Benchmark + Sectors (normalized)")
        norm = sector_prices / sector_prices.iloc[0] * 100.0
        norm_display = norm.copy()
        norm_display = norm_display.rename(columns=lambda c: TICKER_TO_LABEL.get(c, c))
        fig = px.line(norm_display, x=norm_display.index, y=norm_display.columns)
        fig.update_layout(height=460, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Sector 1M Heatmap")
        perf = pd.DataFrame({
            "1W": sector_prices.pct_change(5).iloc[-1] * 100,
            "1M": sector_prices.pct_change(21).iloc[-1] * 100,
            "3M": sector_prices.pct_change(63).iloc[-1] * 100,
            "6M": sector_prices.pct_change(126).iloc[-1] * 100,
            "1Y": sector_prices.pct_change(252).iloc[-1] * 100,
        }).drop(index=[BENCHMARK], errors="ignore").sort_values("1M", ascending=False)

        heat = perf.copy()
        heat.index = [TICKER_TO_LABEL.get(idx, idx) for idx in heat.index]
        fig2 = px.imshow(
            heat.values,
            x=heat.columns,
            y=heat.index,
            aspect="auto",
            color_continuous_scale="RdYlGn",
        )
        # Friendlier hover text: show sector, period, percent with 1 decimal place.
        fig2.update_traces(hovertemplate="%{y} — %{x}: %{z:.1f}%<extra></extra>")
        fig2.update_layout(height=460, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# VIX
# -----------------------------
with tabs[1]:
    st.subheader("VIX Level + Regimes")
    if vix.empty:
        st.info("VIX data is unavailable for the selected period.")
    else:
        v = vix[VIX_TICKER].copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=v.index, y=v.values, name="VIX", mode="lines"))
        for level, label in [(15, "Complacent"), (20, "Normal"), (30, "High"), (40, "Stress")]:
            fig.add_hline(y=level, line_width=1, line_dash="dot", annotation_text=label, annotation_position="top left")
        fig.update_layout(height=520, yaxis_title="VIX")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("VIX vs SPY (rolling 60d correlation)")
        spy_ret = sector_prices[BENCHMARK].pct_change().dropna()
        vix_ret = v.pct_change().dropna()
        joined = pd.concat([spy_ret.rename("SPY"), vix_ret.rename("VIX")], axis=1).dropna()
        corr = joined["SPY"].rolling(60).corr(joined["VIX"])
        figc = px.line(corr, x=corr.index, y=corr.values)
        figc.update_layout(height=380, yaxis_title="Correlation")
        st.plotly_chart(figc, use_container_width=True)

# -----------------------------
# Fear & Greed
# -----------------------------
with tabs[2]:
    st.subheader("Fear & Greed Index")
    if fg_score is None:
        st.warning("Fear & Greed data is unavailable (endpoint may be blocked or changed). The rest of the dashboard still works.")
    else:
        g1, g2 = st.columns([1, 2])
        with g1:
            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fg_score,
                gauge={
                    "axis": {"range": [0, 100]},
                    "steps": [
                        {"range": [0, 25], "color": "#e74c3c"},
                        {"range": [25, 45], "color": "#f39c12"},
                        {"range": [45, 55], "color": "#bdc3c7"},
                        {"range": [55, 75], "color": "#27ae60"},
                        {"range": [75, 100], "color": "#2ecc71"},
                    ],
                },
                title={"text": fg_rating or "Fear & Greed"},
            ))
            fig.update_layout(height=360, margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig, use_container_width=True)
        with g2:
            if fg_hist is not None and not fg_hist.empty:
                fig = px.line(fg_hist, x=fg_hist.index, y="fear_greed")
                fig.update_layout(height=360, yaxis_title="Index (0–100)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Historical Fear & Greed series not available from source.")

# -----------------------------
# Sector Rotation
# -----------------------------
with tabs[3]:
    st.subheader("Sector Rotation (RRG-style proxies)")

    cfg = RotationConfig(z_window=int(rs_z_window), mom_window=int(mom_window))
    rot = compute_sector_rotation(sector_prices.dropna(), benchmark=BENCHMARK, cfg=cfg)
    if rot.empty:
        st.info("Not enough data to compute rotation. Try a longer date range.")
    else:
        # Quadrants
        rot["Quadrant"] = rot.apply(lambda r: quadrant(r["RS_Ratio"], r["RS_Momentum"]), axis=1)

        # Map ticker -> label
        ticker_to_label = {v: k for k, v in SECTOR_ETFS.items()}
        rot["Sector"] = rot["ticker"].map(ticker_to_label).fillna(rot["ticker"])

        # Scatter (RRG proxy)
        fig = px.scatter(
            rot,
            x="RS_Ratio",
            y="RS_Momentum",
            color="Quadrant",
            hover_data=["Sector", "ticker", "Price_1M", "Price_3M", "Price_6M"],
            text="ticker",
        )
        # Quadrant lines at 0,0
        fig.add_hline(y=0, line_width=1, line_dash="dot")
        fig.add_vline(x=0, line_width=1, line_dash="dot")
        fig.update_traces(textposition="top center")
        fig.update_layout(height=560, xaxis_title="RS Ratio (z-score)", yaxis_title=f"RS Momentum (z-score, {mom_window}d)")
        st.plotly_chart(fig, use_container_width=True)

        # Table
        st.subheader("Rotation Table")
        show = rot[["Sector", "ticker", "Quadrant", "RS_Ratio", "RS_Momentum", "Price_1M", "Price_3M", "Price_6M"]].copy()
        show = show.sort_values(["Quadrant", "RS_Ratio"], ascending=[True, False])
        st.dataframe(show, use_container_width=True, hide_index=True)

        # RS ratio time series selector
        st.subheader("Relative Strength (Sector / SPY)")
        sel = st.multiselect("Select sectors", options=list(SECTOR_ETFS.values()), default=["XLK", "XLF", "XLE"])
        if sel:
            ratios = sector_prices[sel].div(sector_prices[BENCHMARK], axis=0)
            ratios_display = ratios.rename(columns=lambda c: TICKER_TO_LABEL.get(c, c))
            fig = px.line(ratios_display, x=ratios_display.index, y=ratios_display.columns)
            fig.update_layout(height=420, yaxis_title="Price Ratio")
            st.plotly_chart(fig, use_container_width=True)

st.caption("Educational dashboard only — not financial advice.")
