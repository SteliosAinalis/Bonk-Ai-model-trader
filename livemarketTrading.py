import os
import time
import math
import ccxt
import joblib
import torch
import numpy as np
import pandas as pd
from collections import deque
import torch.nn as nn

# ---------------- PARAMETERS ---------------- #
DATA_TIMEFRAME = '5m'
SYMBOL = "BONKUSDT"
WINDOW = 30
LOOKAHEAD_K = 6
POLL_SECONDS = 60
MODEL_PATH = "saved_models/final_lstm.pth"
SCALER_PATH = "saved_models/scaler.gz"
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

MAX_RISK_PCT = 0.5
TP_PCT = 0.0025
SL_PCT = -0.002

LONG_THRESHOLD = 0.60
SHORT_THRESHOLD = 0.40

LEVERAGE = 2

scaler = joblib.load(SCALER_PATH)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=n_layers, batch_first=True, dropout=0.1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.classifier(last)

model = LSTMModel(input_size=len(scaler.mean_)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


API_KEY = os.getenv('COINEX_API_KEY')
API_SECRET = os.getenv('COINEX_SECRET')
if not API_KEY or not API_SECRET:
    raise SystemExit("Set COINEX_API_KEY and COINEX_SECRET environment variables.")

exchange = ccxt.coinex({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap',
    },
})


def fetch_ohlcv_df(limit=500):
    ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=DATA_TIMEFRAME, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['ts'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Europe/Athens')
    return df

def rsi(series, length=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(length).mean()
    down = -delta.clip(upper=0).rolling(length).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))

def get_usdt_balance():
    try:
        bal = exchange.fetch_balance()
        return float(bal['free'].get('USDT', 0.0))
    except Exception as e:
        print("Error fetching USDT free balance:", e)
        return 0.0

def place_futures_order(side, qty, leverage=3, limit_price=None, reduce_only=False):
    try:
        exchange.load_markets()
        market = exchange.market(SYMBOL)
        min_qty = float(market.get('limits', {}).get('amount', {}).get('min') or 100000)

        qty = max(qty, 0.0)
        precision = int(market.get('precision', {}).get('amount', 6))
        factor = 10 ** precision
        qty_rounded = math.floor(qty * factor) / factor

        if qty_rounded < min_qty:
            print(f"Order qty {qty_rounded} below min {min_qty}, skipping.")
            return None, 0.0, 0.0

        params = {'marginMode': 'isolated'}

        if reduce_only:
            params['reduceOnly'] = True
        else:
            try:
                exchange.set_leverage(leverage, SYMBOL, {'marginMode': 'isolated'})
                print(f"Set leverage to {leverage}x for {SYMBOL} (isolated).")
            except Exception as e:
                if 'Leverage not changed' in str(e):
                    print(f"Leverage already set to {leverage}x.")
                else:
                    print(f"Could not set leverage: {e}")
                    return None, 0.0, 0.0

        order_type = "market" if limit_price is None else "limit"
        order = exchange.create_order(SYMBOL, order_type, side, qty_rounded, limit_price, params)
        ticker = exchange.fetch_ticker(SYMBOL)
        market_price = float(ticker.get('last') or ticker.get('close') or 0.0)
        executed_price = float(order.get('price') or order.get('average') or market_price)
        executed_qty = float(order.get('filled') or qty_rounded)

        print("Order placed:", side, "qty:", executed_qty, "price:", executed_price)
        return order, executed_price, executed_qty

    except Exception as e:
        print(f"Error placing {side} order:", type(e), e, getattr(e,'args',None))
        return None, 0.0, 0.0

FEATURES = ['return','log_return','ma_fast','ma_slow','ma_ratio','volatility','vol_change','rsi_14']
df = fetch_ohlcv_df(limit=WINDOW + 300).sort_values('timestamp').reset_index(drop=True)
df['return'] = df['close'].pct_change()
df['log_return'] = np.log(df['close']).diff()
df['ma_fast'] = df['close'].rolling(10).mean()
df['ma_slow'] = df['close'].rolling(30).mean()
df['ma_ratio'] = df['ma_fast']/df['ma_slow']
df['volatility'] = df['return'].rolling(30).std()
df['vol_change'] = df['volume'].pct_change()
df['rsi_14'] = rsi(df['close'], 14)
df = df.ffill().bfill().reset_index(drop=True)
feat_buf = deque(df[FEATURES].values[-WINDOW:], maxlen=WINDOW)
last_candle_ts = df['timestamp'].iloc[-1]

position = 0.0
entry_price = None
position_type = None
position_value = 0.0

print("Starting LIVE trading.")

while True:
    try:
        latest = fetch_ohlcv_df(limit=5).sort_values('timestamp').reset_index(drop=True)
        new_candles = latest[latest['timestamp'] > last_candle_ts]
        if new_candles.empty:
            new_candles = latest.iloc[[-1]]

        df = pd.concat([df, new_candles], ignore_index=True)
        df['return'] = df['close'].pct_change()
        df['log_return'] = np.log(df['close']).diff()
        df['ma_fast'] = df['close'].rolling(10).mean()
        df['ma_slow'] = df['close'].rolling(30).mean()
        df['ma_ratio'] = df['ma_fast']/df['ma_slow']
        df['volatility'] = df['return'].rolling(30).std()
        df['vol_change'] = df['volume'].pct_change()
        df['rsi_14'] = rsi(df['close'], 14)
        df = df.ffill().bfill().reset_index(drop=True)

        last_feats = df[FEATURES].values[-WINDOW:]
        X_df = pd.DataFrame(last_feats, columns=FEATURES).replace([np.inf,-np.inf], np.nan).ffill().fillna(0.0)
        Xs = scaler.transform(X_df).reshape(1, WINDOW, len(FEATURES))
        Xt = torch.tensor(Xs, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            prob = float(model(Xt).cpu().numpy().flatten()[0])

        row = new_candles.iloc[-1]
        price = float(row['close'])
        usdt_free = get_usdt_balance()
        print(f"{row['timestamp']} model prob {prob:.4f} USDT_free ${usdt_free:.2f} position={position_type} qty={position}")

        #close positions
        if position > 0 and position_type is not None:
            if position_type == "long":
                tp_price = entry_price * (1 + TP_PCT)
                sl_price = entry_price * (1 + SL_PCT)
                if row['low'] <= sl_price or row['high'] >= tp_price:
                    order, executed_price, executed_qty = place_futures_order("sell", position, reduce_only=True)
                    if executed_price:
                        pnl = (executed_price - entry_price) * executed_qty
                        print(f"Closed LONG entry={entry_price:.8f} exit={executed_price:.8f} pnl={pnl:.6f}")
                    position, entry_price, position_type, position_value = 0.0, None, None, 0.0

            elif position_type == "short":
                tp_price = entry_price * (1 - TP_PCT)
                sl_price = entry_price * (1 - SL_PCT)
                if row['high'] >= sl_price or row['low'] <= tp_price:
                    order, executed_price, executed_qty = place_futures_order("buy", position, reduce_only=True)
                    if executed_price:
                        pnl = (entry_price - executed_price) * executed_qty
                        print(f"Closed SHORT entry={entry_price:.8f} exit={executed_price:.8f} pnl={pnl:.6f}")
                    position, entry_price, position_type, position_value = 0.0, None, None, 0.0

        #open positions
        usd_to_use = usdt_free * MAX_RISK_PCT if usdt_free > 0 else 0.0
        if usd_to_use > 10 and position == 0:
            
            qty = (usd_to_use * LEVERAGE) / price
            min_qty = 100000
            try:
                market = exchange.market(SYMBOL)
                min_qty = float(market.get('limits', {}).get('amount', {}).get('min') or 100000)
            except:
                pass

            if qty < min_qty:
                print(f"Calculated qty {qty:.0f} below minimum {min_qty}, skipping.")
                qty = 0.0

            if qty >= min_qty:
                if prob >= LONG_THRESHOLD:
                    order, executed_price, executed_qty = place_futures_order("buy", qty, leverage=LEVERAGE)
                    if executed_price:
                        entry_price = executed_price
                        position, position_type, position_value = executed_qty, "long", usd_to_use
                        print(f"Opened LIVE LONG qty={position:.0f} entry_price={entry_price:.8f}")

                elif prob <= SHORT_THRESHOLD:
                    order, executed_price, executed_qty = place_futures_order("sell", qty, leverage=LEVERAGE)
                    if executed_price:
                        entry_price = executed_price
                        position, position_type, position_value = executed_qty, "short", usd_to_use
                        print(f"Opened LIVE SHORT qty={position:.0f} entry_price={entry_price:.8f}")

        #trainnig
        if len(df) > WINDOW + LOOKAHEAD_K:
            future_price = df['close'].iloc[-LOOKAHEAD_K]
            label = 1.0 if future_price > price else 0.0
            y_true = torch.tensor([[label]], dtype=torch.float32).to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(Xt)
            loss = nn.BCELoss()(y_pred, y_true)
            loss.backward()
            optimizer.step()

        last_candle_ts = new_candles['timestamp'].iloc[-1]
        time.sleep(POLL_SECONDS)

    except KeyboardInterrupt:
        print("\nBot stopped manually. Closing open positions if any...")
        if position > 0:
            side = "sell" if position_type == "long" else "buy"
            place_futures_order(side, position, reduce_only=True)
            print("Position closed.")
        break
    except Exception as e:
        print(f"Exception in main loop: {type(e).__name__} - {e}")
        time.sleep(5)