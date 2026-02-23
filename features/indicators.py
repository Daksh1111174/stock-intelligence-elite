import pandas as pd

def add_indicators(df):

    df["Daily_Return"] = df["Close"].pct_change()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["Volatility"] = df["Daily_Return"].rolling(20).std()

    return df.dropna()
