import yfinance as yf
import pandas as pd

def load_stock(ticker, start, end):

    df = yf.download(ticker, start=start, end=end, auto_adjust=True)

    if df.empty:
        return df

    # Fix MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)

    return df


def load_portfolio(tickers, start, end):

    df = yf.download(tickers, start=start, end=end)

    if df.empty:
        return df

    # Keep only Close
    df = df["Close"]

    return df
