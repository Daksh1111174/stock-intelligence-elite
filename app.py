import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time

from data.loader import load_stock, load_portfolio
from features.indicators import add_indicators
from models.regression import train_model
from models.shap_explain import shap_plot
from portfolio.optimizer import efficient_frontier
from utils.signals import trading_signal

st.set_page_config(layout="wide")

# Load Glass UI
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🚀 Stock Intelligence Elite")

ticker = st.sidebar.text_input("Ticker", "AAPL")
start = st.sidebar.date_input("Start Date")
end = st.sidebar.date_input("End Date")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Live Market",
    "⚡ Signals",
    "📊 Optimizer",
    "🧠 Explainability",
    "🌌 Correlation 3D"
])

# ---------- TAB 1 LIVE ----------
with tab1:

    placeholder = st.empty()

    for _ in range(5):
        live = load_stock(ticker, start, end)
        latest_price = live["Close"].iloc[-1]

        with placeholder.container():
            st.metric("Live Price", round(latest_price,2))

        time.sleep(2)

# ---------- LOAD DATA ----------
df = load_stock(ticker, start, end)
df = add_indicators(df)

# ---------- MODEL ----------
model, rmse, X_test = train_model(df)

# ---------- TAB 2 SIGNAL ----------
with tab2:
    signal = trading_signal(df["SMA_20"].iloc[-1], df["SMA_50"].iloc[-1])

    st.markdown(f"""
    <h2 style='text-align:center;
    animation: pulse 1.5s infinite;'>
    {signal}
    </h2>
    """, unsafe_allow_html=True)

# ---------- TAB 3 OPTIMIZER ----------
with tab3:

    tickers = ["AAPL","MSFT","GOOG","TSLA"]
    portfolio_data = load_portfolio(tickers, start, end)
    returns = portfolio_data.pct_change().dropna()

    results = efficient_frontier(returns)

    fig = go.Figure(data=go.Scatter(
        x=results[:,1],
        y=results[:,0],
        mode="markers",
        marker=dict(
            color=results[:,2],
            colorscale="Viridis",
            showscale=True
        )
    ))

    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# ---------- TAB 4 SHAP ----------
with tab4:
    st.pyplot(shap_plot(model, X_test))

# ---------- TAB 5 3D CORRELATION ----------
with tab5:

    corr = portfolio_data.corr()

    fig = go.Figure(data=[go.Surface(
        z=corr.values,
        x=corr.columns,
        y=corr.columns
    )])

    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
