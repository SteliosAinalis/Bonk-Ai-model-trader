# FILE: liveMarketPaperTrading.py (FINAL - Original Polling Logic Restored)
import os
import time
import ccxt
import joblib
import torch
import numpy as np
import pandas as pd
from collections import deque
import torch.nn as nn

# --- SCRIPT PARAMETERS ---
DATA_TIMEFRAME = '5m'
SYMBOL = 'BONKUSDT'
WINDOW = 30
LOOKAHEAD_K = 6
POLL_SECONDS = 20
MODEL_PATH = "saved_models/best_lstm.pth"
SCALER_PATH = "saved_models/scaler.gz"
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
TIMEZONE = 'Europe/Athens'

# --- TRADING STRATEGY PARAMETERS ---
MAX_RISK_PCT = 1.0 # As per your design
TP_PCT = 0.005
SL_PCT = -0.003
FEE_PCT = 0.001
SLIPPAGE_PCT = 0.0005
LONG_THRESHOLD = 0.53
SHORT_THRESHOLD = 0.47

# --- LOAD MODEL AND SCALER ---
scaler = joblib.load(SCALER_PATH)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=n_layers, batch_first=True, dropout=0.1)
        self.classifier = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
    def forward(self, x):
        out, _ = self.lstm(x); last = out[:, -1, :]; return self.classifier(last)

model = LSTMModel(input_size=len(scaler.mean_)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.train() 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.BCELoss()

# --- CONNECT TO EXCHANGE ---
API_KEY = os.getenv('COINEX_API_KEY')
API_SECRET = os.getenv('COINEX_SECRET')
if not API_KEY or not API_SECRET: raise SystemExit("Set COINEX_API_KEY and COINEX_SECRET env variables.")
exchange = ccxt.coinex({'apiKey': API_KEY, 'secret': API_SECRET, 'enableRateLimit': True})

# --- UTILITY FUNCTIONS ---
def fetch_ohlcv_df(limit=500):
    ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=DATA_TIMEFRAME, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['ts'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(TIMEZONE)
    return df

FEATURES = ['return', 'log_return', 'ma_fast', 'ma_slow', 'ma_ratio', 'volatility', 'vol_change', 'rsi_14']
def rsi(series, length=14):
    delta = series.diff(); up = delta.clip(lower=0).rolling(length).mean(); down = -delta.clip(upper=0).rolling(length).mean()
    rs = up / (down + 1e-12); return 100 - (100 / (1 + rs))

# --- INITIALIZE DATA AND STATE ---
print("Initializing data buffer...")
df = fetch_ohlcv_df(limit=WINDOW + 300).sort_values('timestamp').reset_index(drop=True)
# (Feature calculation logic is correct)
df['return'] = df['close'].pct_change(); df['log_return'] = np.log(df['close']).diff()
df['ma_fast'] = df['close'].rolling(10).mean(); df['ma_slow'] = df['close'].rolling(30).mean()
df['ma_ratio'] = df['ma_fast'] / df['ma_slow']; df['volatility'] = df['return'].rolling(30).std()
df['vol_change'] = df['volume'].pct_change(); df['rsi_14'] = rsi(df['close'], 14)
df = df.ffill().bfill().reset_index(drop=True)

last_candle_ts = df['timestamp'].iloc[-1]
paper_balance = 1000.0; position = 0.0; entry_price = None
position_type = None; position_value = 0.0
prediction_history = deque(maxlen=LOOKAHEAD_K + 5)

print(f"Paper trading initialized with ${paper_balance:.2f}. Starting live loop...")
while True:
    try:
        # --- CONTINUOUS POLLING LOGIC ---
        latest = fetch_ohlcv_df(limit=5).sort_values('timestamp').reset_index(drop=True)
        new_candles = latest[latest['timestamp'] > last_candle_ts]

        # The dataframe is always updated with the latest info
        df = pd.concat([df, new_candles], ignore_index=True).drop_duplicates(subset=['timestamp'], keep='last')
        
        # Re-calculate features on every loop
        df['return'] = df['close'].pct_change(); df['log_return'] = np.log(df['close']).diff()
        df['ma_fast'] = df['close'].rolling(10).mean(); df['ma_slow'] = df['close'].rolling(30).mean()
        df['ma_ratio'] = df['ma_fast'] / df['ma_slow']; df['volatility'] = df['return'].rolling(30).std()
        df['vol_change'] = df['volume'].pct_change(); df['rsi_14'] = rsi(df['close'], 14)
        df = df.ffill().bfill()
        
        # Use the most recent row for checks, even if it's an incomplete candle
        current_row = df.iloc[-1]
        
        # --- POSITION MANAGEMENT ---
        if position > 0:
            closed = False; exit_price = None; reason = None
            if position_type == "long":
                tp_price = entry_price * (1 + TP_PCT); sl_price = entry_price * (1 + SL_PCT)
                if current_row['low'] <= sl_price: exit_price = sl_price; reason = 'stop_loss'; closed = True
                elif current_row['high'] >= tp_price: exit_price = tp_price; reason = 'take_profit'; closed = True
            elif position_type == "short":
                tp_price = entry_price * (1 - TP_PCT); sl_price = entry_price * (1 + SL_PCT)
                if current_row['high'] >= sl_price: exit_price = sl_price; reason = 'stop_loss'; closed = True
                elif current_row['low'] <= tp_price: exit_price = tp_price; reason = 'take_profit'; closed = True

            if closed:
                pnl = (exit_price - entry_price) * position if position_type == "long" else (entry_price - exit_price) * position
                paper_balance += position_value + pnl
                print(f"  >>> Closed {position_type.upper()} | Reason: {reason} | PnL: ${pnl:.4f} | New Balance: ${paper_balance:.2f}")
                position, entry_price, position_type, position_value = 0.0, None, None, 0.0

        # --- MODEL PREDICTION & ONLINE LEARNING ---
        last_feats = df[FEATURES].values[-WINDOW:]
        X_df = pd.DataFrame(last_feats, columns=FEATURES).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
        Xs = scaler.transform(X_df).reshape(1, WINDOW, len(FEATURES))
        Xt = torch.tensor(Xs, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            prob = float(model(Xt).cpu().numpy().flatten()[0])
        
        prediction_history.append({'features': Xt, 'price_at_prediction': df['close'].iloc[-1]})
        if len(prediction_history) > LOOKAHEAD_K:
            old_prediction = prediction_history.popleft()
            ground_truth_price = df['close'].iloc[-1]
            y_true_label = 1.0 if ground_truth_price > old_prediction['price_at_prediction'] else 0.0
            y_true = torch.tensor([[y_true_label]], dtype=torch.float32).to(DEVICE)
            optimizer.zero_grad(); y_pred = model(old_prediction['features']); loss = criterion(y_pred, y_true)
            loss.backward(); optimizer.step()

        # --- TRADE ENTRY ---
        price = float(current_row['close'])
        print(f"{current_row['timestamp']} | Prob: {prob:.4f} | Balance: ${paper_balance:.2f} | Pos: {position_type or 'None'}")
        if position == 0 and paper_balance > 0:
            usd_to_use = paper_balance * MAX_RISK_PCT
            if prob >= LONG_THRESHOLD:
                qty = usd_to_use / price; entry_price = price * (1 + SLIPPAGE_PCT + FEE_PCT)
                position, position_type, position_value = qty, "long", usd_to_use
                paper_balance -= usd_to_use
                print(f"  <<< Opened LONG | Qty: {qty:.4f} at ~{price:.6f}")
            elif prob <= SHORT_THRESHOLD:
                qty = usd_to_use / price; entry_price = price * (1 - SLIPPAGE_PCT - FEE_PCT)
                position, position_type, position_value = qty, "short", usd_to_use
                paper_balance -= usd_to_use
                print(f"  <<< Opened SHORT | Qty: {qty:.4f} at ~{price:.6f}")

        # Update last candle timestamp only if a new full candle was received
        if not new_candles.empty:
            last_candle_ts = new_candles.iloc[-1]['timestamp']
        
        time.sleep(POLL_SECONDS)
    
    except KeyboardInterrupt: print("\nBot stopped manually."); break
    except Exception as e:
        print(f"An error occurred: {type(e).__name__} - {e}. Restarting loop in 60s...")
        time.sleep(60)