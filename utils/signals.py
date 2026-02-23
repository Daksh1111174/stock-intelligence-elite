def trading_signal(short_ma, long_ma):
    if short_ma > long_ma:
        return "🟢 BUY SIGNAL"
    else:
        return "🔴 SELL SIGNAL"
