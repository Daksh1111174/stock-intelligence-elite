import streamlit as st
import yfinance as yf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime
import sys
import os

# Fix path for Streamlit Cloud
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.loader import load_stock, load_portfolio
from features.indicators import add_indicators
from features.regime_detection import detect_nifty_regime
from models.regression import train_regression
from portfolio.optimizer import efficient_frontier
from utils.signals import trading_signal

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="🇮🇳 Stock Intelligence Elite",
    layout="wide"
)

# Load Glass UI
try:
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except:
    pass

st.title("🇮🇳 Stock Intelligence Elite – NSE Edition")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.header("⚙ Configuration")

ticker = st.sidebar.selectbox(
    "Select NSE Stock",
    ["RELIANCE.NS", "TCS.NS", "INFY.NS",
     "HDFCBANK.NS", "ICICIBANK.NS"]
)

start = st.sidebar.date_input("Start Date")
end = st.sidebar.date_input("End Date")

run = st.sidebar.button("Run Analysis")

# ---------------------------------------------------
# TABS
# ---------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📡 Live Market",
    "⚡ Trading Signals",
    "📊 Portfolio Optimizer",
    "🧠 ML Insights",
    "🌌 Correlation 3D",
    "📉 NIFTY 50 Regime"
])

# ---------------------------------------------------
# TAB 1 — LIVE MARKET (ROBUST VERSION)
# ---------------------------------------------------
with tab1:

    st.subheader("📡 Live Market Data")

    now = datetime.datetime.now()

    if now.weekday() >= 5:
        st.info("Market Closed (Weekend)")

    try:
        # Try 1-minute interval
        live = yf.download(ticker, period="1d", interval="1m")

        if live.empty:
            # Fallback to 5-minute
            live = yf.download(ticker, period="1d", interval="5m")

        if live.empty:
            # Final fallback daily
            daily = yf.download(ticker, period="5d", interval="1d")

            if daily.empty:
                st.error("Market data unavailable.")
            else:
                latest_price = daily["Close"].iloc[-1]
                st.metric("Latest Close Price", round(latest_price, 2))
                st.info("Showing latest available daily close.")
        else:
            latest_price = live["Close"].iloc[-1]
            st.metric("Live Price", round(latest_price, 2))

            fig = px.line(
                live,
                x=live.index,
                y="Close",
                template="plotly_dark",
                title="Intraday Price"
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception:
        st.warning("Live intraday data unavailable. Showing last close.")

        fallback = yf.download(ticker, period="5d", interval="1d")

        if not fallback.empty:
            st.metric("Latest Close Price",
                      round(fallback["Close"].iloc[-1], 2))

# ---------------------------------------------------
# MAIN ANALYSIS
# ---------------------------------------------------
if run:

    df = load_stock(ticker, start, end)

    if df.empty:
        st.error("No historical data available.")
        st.stop()

    df = add_indicators(df)

    if df.empty:
        st.error("Not enough data to compute indicators.")
        st.stop()

    # ---------------- TAB 2 SIGNALS ----------------
    with tab2:

        st.subheader("📊 Moving Average Strategy")

        signal = trading_signal(
            df["SMA_20"].iloc[-1],
            df["SMA_50"].iloc[-1]
        )

        st.markdown(f"## {signal}")

        fig = px.line(
            df,
            x="Date",
            y="Close",
            template="plotly_dark",
            title="Historical Price"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- TAB 3 PORTFOLIO ----------------
    with tab3:

        st.subheader("📈 Efficient Frontier")

        tickers = ["RELIANCE.NS", "TCS.NS",
                   "INFY.NS", "HDFCBANK.NS",
                   "ICICIBANK.NS"]

        portfolio_data = load_portfolio(tickers, start, end)
        returns = portfolio_data.pct_change().dropna()

        if returns.empty:
            st.warning("Not enough data for portfolio optimization.")
        else:
            results = efficient_frontier(returns)

            fig = go.Figure(data=go.Scatter(
                x=results[:, 1],
                y=results[:, 0],
                mode="markers",
                marker=dict(
                    color=results[:, 2],
                    colorscale="Viridis",
                    showscale=True
                )
            ))

            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Volatility",
                yaxis_title="Expected Return"
            )

            st.plotly_chart(fig, use_container_width=True)

    # ---------------- TAB 4 ML ----------------
    with tab4:

        st.subheader("📊 Regression Model")

        model, rmse, X_test = train_regression(df)

        st.metric("Random Forest RMSE", round(rmse, 2))

        importance = model.feature_importances_

        fig = px.bar(
            x=importance,
            y=X_test.columns,
            orientation="h",
            template="plotly_dark",
            title="Feature Importance"
        )

        st.plotly_chart(fig, use_container_width=True)

    # ---------------- TAB 5 CORRELATION ----------------
    with tab5:

        st.subheader("🌌 3D Correlation Matrix")

        corr = portfolio_data.corr()

        fig = go.Figure(data=[go.Surface(
            z=corr.values,
            x=corr.columns,
            y=corr.columns
        )])

        fig.update_layout(template="plotly_dark")

        st.plotly_chart(fig, use_container_width=True)

    # ---------------- TAB 6 REGIME DETECTION ----------------
    with tab6:

        st.subheader("📉 NIFTY 50 Regime Detection")

        regime_df = detect_nifty_regime(start, end)

        if regime_df.empty:
            st.warning("Not enough data for regime detection.")
        else:
            fig = px.scatter(
                regime_df,
                x=regime_df.index,
                y="Close",
                color="Regime",
                template="plotly_dark",
                title="NIFTY 50 Market Regimes"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Regime Distribution")
            st.write(regime_df["Regime"].value_counts())
