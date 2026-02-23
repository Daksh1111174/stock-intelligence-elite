import yfinance as yf

def load_stock(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df.reset_index(inplace=True)
    return df

def load_portfolio(tickers, start, end):
    df = yf.download(tickers, start=start, end=end)["Close"]
    return df
