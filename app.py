import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from spark.spark_session import create_spark
from data.loader import load_stock, load_portfolio
from features.indicators import add_indicators
from models.regression import train_regression
from models.lstm import train_lstm
from models.shap_explain import shap_plot
from portfolio.optimizer import efficient_frontier
from utils.signals import trading_signal

st.set_page_config(layout="wide")

with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🇮🇳 Stock Intelligence Elite – NSE")

ticker = st.sidebar.selectbox(
    "Select NSE Stock",
    ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS"]
)

start = st.sidebar.date_input("Start Date")
end = st.sidebar.date_input("End Date")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📡 Live Market",
    "⚡ Signals",
    "📊 Portfolio",
    "🧠 ML Explainability",
    "🌌 Correlation 3D"
])

# Live Market
with tab1:
    live = yf.download(ticker, period="1d", interval="1m")
    if not live.empty:
        st.metric("Live Price", round(live["Close"].iloc[-1],2))
        fig = px.line(live, x=live.index, y="Close", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# Historical
df = load_stock(ticker, start, end)
df = add_indicators(df)

# Regression
model, rmse, X_test = train_regression(df)

# Signals
with tab2:
    signal = trading_signal(df["SMA_20"].iloc[-1], df["SMA_50"].iloc[-1])
    st.markdown(f"## {signal}")

# Portfolio
with tab3:
    tickers = ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS"]
    portfolio_data = load_portfolio(tickers, start, end)
    returns = portfolio_data.pct_change().dropna()
    results = efficient_frontier(returns)

    fig = go.Figure(data=go.Scatter(
        x=results[:,1],
        y=results[:,0],
        mode="markers",
        marker=dict(color=results[:,2], colorscale="Viridis", showscale=True)
    ))
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig)

# SHAP
with tab4:
    st.pyplot(shap_plot(model, X_test))

# 3D Correlation
with tab5:
    corr = portfolio_data.corr()
    fig = go.Figure(data=[go.Surface(z=corr.values)])
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig)
