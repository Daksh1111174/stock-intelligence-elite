import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import time

from data.loader import load_stock, load_portfolio
from features.indicators import add_indicators
from models.regression import train_model
from models.feature_importance import feature_importance_plot
from portfolio.optimizer import efficient_frontier
from utils.signals import trading_signal

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Stock Intelligence Elite",
    layout="wide"
)

# Load Glass UI
try:
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except:
    pass

st.title("🚀 Stock Intelligence Elite")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.header("⚙ Configuration")

ticker = st.sidebar.text_input("Ticker", "AAPL")
start = st.sidebar.date_input("Start Date")
end = st.sidebar.date_input("End Date")

run = st.sidebar.button("Run Analysis")

# ---------------------------------------------------
# TABS
# ---------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Live Market",
    "⚡ Trading Signals",
    "📊 Portfolio Optimizer",
    "🧠 Model Insights",
    "🌌 Correlation 3D"
])

# ---------------------------------------------------
# TAB 1 — LIVE MARKET (SAFE VERSION)
# ---------------------------------------------------
with tab1:

    st.subheader("📡 Live Price Feed")

    try:
        live = yf.download(ticker, period="1d", interval="1m")

        if live.empty:
            st.warning("Live data unavailable (market closed or API limit).")
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
        st.error("Live data fetch failed. Try again later.")

# ---------------------------------------------------
# MAIN ANALYSIS
# ---------------------------------------------------
if run:

    try:
        df = load_stock(ticker, start, end)

        if df.empty:
            st.error("No historical data available for selected range.")
            st.stop()

        df = add_indicators(df)

        if df.empty:
            st.error("Not enough data to compute indicators.")
            st.stop()

    except Exception:
        st.error("Data loading failed.")
        st.stop()

    # ---------------- MODEL ----------------
    try:
        model, rmse, X_test = train_model(df)
    except Exception:
        st.error("Model training failed.")
        st.stop()

    # ---------------- TAB 2 SIGNALS ----------------
    with tab2:

        st.subheader("📊 Moving Average Strategy")

        signal = trading_signal(
            df["SMA_20"].iloc[-1],
            df["SMA_50"].iloc[-1]
        )

        st.markdown(
            f"<h2 style='text-align:center;'>{signal}</h2>",
            unsafe_allow_html=True
        )

    # ---------------- TAB 3 OPTIMIZER ----------------
    with tab3:

        st.subheader("📈 Efficient Frontier")

        tickers = ["AAPL", "MSFT", "GOOG", "TSLA"]

        try:
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

        except Exception:
            st.error("Portfolio optimization failed.")

    # ---------------- TAB 4 MODEL INSIGHTS ----------------
    with tab4:

        st.subheader("📊 Feature Importance")

        try:
            fig = feature_importance_plot(model, X_test)
            st.plotly_chart(fig, use_container_width=True)
            st.metric("Random Forest RMSE", round(rmse, 2))
        except Exception:
            st.error("Feature importance calculation failed.")

    # ---------------- TAB 5 CORRELATION ----------------
    with tab5:

        st.subheader("🌌 3D Correlation Matrix")

        try:
            corr = portfolio_data.corr()

            fig = go.Figure(data=[go.Surface(
                z=corr.values,
                x=corr.columns,
                y=corr.columns
            )])

            fig.update_layout(template="plotly_dark")

            st.plotly_chart(fig, use_container_width=True)

        except Exception:
            st.warning("Correlation visualization unavailable.")
