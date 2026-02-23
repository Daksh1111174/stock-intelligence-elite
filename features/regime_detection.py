import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def detect_nifty_regime(start, end):

    df = yf.download("^NSEI", start=start, end=end)
    df = df[["Close"]].copy()

    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(20).std()

    df = df.dropna()

    # Features for clustering
    X = df[["Return", "Volatility"]]

    kmeans = KMeans(n_clusters=3, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X)

    # Map clusters to regime labels
    cluster_vol = df.groupby("Cluster")["Volatility"].mean()
    sorted_clusters = cluster_vol.sort_values()

    regime_map = {
        sorted_clusters.index[0]: "Low Volatility (Bull)",
        sorted_clusters.index[1]: "Medium Regime",
        sorted_clusters.index[2]: "High Volatility (Bear)"
    }

    df["Regime"] = df["Cluster"].map(regime_map)

    return df
