import ccxt
import pandas as pd
import time
import os
from dotenv import load_dotenv


SYMBOL = 'BTCUSDT'  #for bonk: 1000BONK/USDT    for btc: BTCUSDT
TIMEFRAME = '5m'
START_DATE = '2024-01-01T00:00:00Z' 
OUTPUT_FILENAME = 'btc_usdt_BINANCE_history.csv'  #for bonk: bonk_usdt_BINANCE_history.csv for btc: btc_usdt_BINANCE_history.csv


exchange = ccxt.binance({
    'options': {
        'defaultType': 'future', 
    },
})


print(f"Starting historical data fetch for {SYMBOL} from {START_DATE} on Binance...")

since_timestamp = exchange.parse8601(START_DATE)
all_ohlcv = []
end_timestamp = exchange.milliseconds()
previous_oldest_timestamp = None

while True:
    try:
        print(f"  Fetching 1000 candles ending around {pd.to_datetime(end_timestamp, unit='ms')}...")
        ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=1000, params={'until': end_timestamp})
        if not ohlcv or len(ohlcv) == 0:
            print("No more data available from the exchange."); break
        oldest_timestamp = ohlcv[0][0]
        if oldest_timestamp == previous_oldest_timestamp:
            print("Reached the beginning of the available history. Loop finished."); break
        all_ohlcv = ohlcv + all_ohlcv
        print(f"    Fetched {len(ohlcv)} candles. Oldest is {pd.to_datetime(oldest_timestamp, unit='ms')}. Total: {len(all_ohlcv)}")
        previous_oldest_timestamp = oldest_timestamp
        end_timestamp = oldest_timestamp - 1
        if oldest_timestamp < since_timestamp:
            print("Reached data older than the specified start date. Loop finished."); break
        time.sleep(exchange.rateLimit / 1000)
    except Exception as e:
        print(f"An error occurred: {e}. Stopping."); break

if all_ohlcv:
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.sort_values('timestamp', inplace=True)
    df.drop_duplicates(subset='timestamp', keep='last', inplace=True)
    start_date_naive = pd.to_datetime(START_DATE).tz_localize(None)
    df = df[df['timestamp'] >= start_date_naive]
    
    # --- IMPORTANT NOTE ON PRICE ADJUSTMENT ---
    # The 'open', 'high', 'low', 'close' prices in this file will be 1000x larger
    # than the actual BONK price. The 'volume' will be 1000x smaller.
    # For our AI, which learns from PERCENTAGE CHANGES, this does not matter.
    # The patterns will be identical. We do not need to adjust the prices.
    
    os.makedirs('data', exist_ok=True)
    output_path = os.path.join('data', OUTPUT_FILENAME)
    df.to_csv(output_path, index=False)
    print(f"\nSuccessfully saved {len(df)} candles to {output_path}")
else:
    print("\nNo data was fetched. Nothing to save.")