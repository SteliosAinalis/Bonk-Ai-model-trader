import ccxt
import pandas as pd
import time

exchange = ccxt.binance()
symbol = 'BONK/USDT'
timeframe = '5m'


start_date = int(pd.Timestamp("2025-04-01 00:00:00").timestamp() * 1000)
end_date   = int(pd.Timestamp("2025-08-30 23:59:59").timestamp() * 1000)

all_ohlcv = []
since = start_date
limit = 1000 

while since < end_date:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
    if not ohlcv:
        break
    all_ohlcv += ohlcv
    since = ohlcv[-1][0] + 1  # move to next candle
    time.sleep(exchange.rateLimit / 1000)  # respect rate limit

# Convert to DataFrame
df = pd.DataFrame(all_ohlcv, columns=['timestamp','open','high','low','close','volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

df.to_csv('bonk_usdt_5m_sept_oct_nov.csv', index=False)
print("Saved:", df.shape, "candles")
