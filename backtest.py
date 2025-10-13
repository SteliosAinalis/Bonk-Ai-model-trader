# FILE: backtest.py (FINAL - RELATIONAL FEATURES IMPLEMENTED)
import os
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import joblib
import random
import argparse

# --- PARAMETERS ---
DATA_PATH_BONK = "data/bonk_usdt_BINANCE_history.csv"
DATA_PATH_BTC = "data/btc_usdt_BINANCE_history.csv" # --- NEW: Path for BTC data ---
WINDOW = 30
LOOKAHEAD_K = 6
TEST_FRAC = 0.3
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
MODEL_PATH = "saved_models"
os.makedirs(MODEL_PATH, exist_ok=True)
BEST_MODEL_FILE = os.path.join(MODEL_PATH, "best_lstm.pth")

# --- BACKTESTING PARAMS ---
START_BALANCE = 1000.0
MAX_RISK_PCT = 1.0 # Your intentional 100% risk strategy
STOP_LOSS_PCT = -0.03
TAKE_PROFIT_PCT = 0.05
SLIPPAGE_MIN = 0.0003    
SLIPPAGE_MAX = 0.001     
TAKER_FEE_PCT = 0.00075
MAX_HOLD_PERIOD = 12
COOLDOWN_ON_LOSS = 3
MAX_DRAWDOWN_PCT = 0.5  

# --- UTILITIES ---
def rsi(series, length=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(length).mean()
    down = -delta.clip(upper=0).rolling(length).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))

def max_drawdown(series):
    peak = series.cummax()
    dd = (series - peak) / (peak + 1e-12)
    return dd.min()

def sharpe_ratio(returns, periods_per_year):
    mean_r = np.nanmean(returns)
    std_r = np.nanstd(returns, ddof=1)
    if std_r == 0: return np.nan
    return (mean_r / std_r) * np.sqrt(periods_per_year)
    
# --- DATASET CLASS ---
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, window):
        self.X = X; self.y = y; self.window = window
    def __len__(self):
        return len(self.X) - self.window
    def __getitem__(self, idx):
        x_sequence = self.X[idx : idx + self.window]
        y_target = self.y[idx + self.window - 1]
        return (torch.tensor(x_sequence, dtype=torch.float32),
                torch.tensor(y_target, dtype=torch.float32).unsqueeze(0))

# --- MODEL CLASS ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, n_layers=2):
        super().__init__()
        # --- FIX: Increased dropout for better regularization against overfitting ---
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=n_layers, batch_first=True, dropout=0.2)
        self.classifier = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(),
                                        nn.Linear(32, 1), nn.Sigmoid())
    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.classifier(last)

# --- SCRIPT EXECUTION ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and/or backtest a trading model.")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'backtest'],
                        help="'train' to run training and then backtest. 'backtest' to only run the backtest on a saved model.")
    args = parser.parse_args()

    # -----------------------------
    # LOAD & PREP DATA (NEW VERSION)
    # -----------------------------
    print("Loading data...")
    df_bonk = pd.read_csv(DATA_PATH_BONK)
    df_btc = pd.read_csv(DATA_PATH_BTC)

    df_bonk['timestamp'] = pd.to_datetime(df_bonk['timestamp'])
    df_btc['timestamp'] = pd.to_datetime(df_btc['timestamp'])
    df_bonk.set_index('timestamp', inplace=True)
    df_btc.set_index('timestamp', inplace=True)

    print("Merging BONK and BTC data...")
    df = pd.merge(df_bonk, df_btc, left_index=True, right_index=True, suffixes=('_bonk', '_btc'))
    df = df.reset_index()

    print("Engineering base and relational features...")
    # --- Base BONK features ---
    df['return'] = df['close_bonk'].pct_change()
    df['log_return'] = np.log(df['close_bonk']).diff()
    df['ma_fast'] = df['close_bonk'].rolling(10).mean()
    df['ma_slow'] = df['close_bonk'].rolling(30).mean()
    df['ma_ratio'] = df['ma_fast'] / df['ma_slow']
    df['volatility'] = df['return'].rolling(30).std()
    df['vol_change'] = df['volume_bonk'].pct_change()
    df['rsi_14'] = rsi(df['close_bonk'], 14)

    # --- Relational Features (The Potential Edge) ---
    df['btc_return'] = df['close_btc'].pct_change()
    for i in range(1, 4):
        df[f'btc_return_lag_{i}'] = df['btc_return'].shift(i)
    df['bonk_btc_ratio'] = df['close_bonk'] / df['close_btc']
    df['bonk_btc_ratio_ema_20'] = df['bonk_btc_ratio'].ewm(span=20, adjust=False).mean()
    df['bonk_vs_btc_momentum'] = df['bonk_btc_ratio'] / df['bonk_btc_ratio_ema_20']
    df['btc_atr'] = (df['high_btc'] - df['low_btc']).rolling(14).mean()
    df['market_vol_ema_50'] = df['btc_atr'].ewm(span=50, adjust=False).mean()
    df['market_vol_expansion'] = df['btc_atr'] / (df['market_vol_ema_50'] + 1e-12)

    df['future_return'] = df['close_bonk'].shift(-LOOKAHEAD_K) / df['close_bonk'] - 1
    df['target'] = (df['future_return'] > 0).astype(int)

    FEATURES = [
        'return','log_return','ma_fast','ma_slow','ma_ratio','volatility','vol_change','rsi_14',
        'btc_return_lag_1', 'btc_return_lag_2', 'btc_return_lag_3',
        'bonk_vs_btc_momentum', 'market_vol_expansion'
    ]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.fillna(0, inplace=True)
    df = df.reset_index(drop=True)
    
    split_idx = int(len(df) * (1 - TEST_FRAC))
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(df_train[FEATURES])
    y_train_flat = df_train['target'].values
    X_test_flat = scaler.transform(df_test[FEATURES])
    y_test_flat = df_test['target'].values
    joblib.dump(scaler, os.path.join(MODEL_PATH, "scaler.gz"))
    
    test_dataset = TimeSeriesDataset(X_test_flat, y_test_flat, WINDOW)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = LSTMModel(len(FEATURES)).to(DEVICE)
    
    if args.mode == 'train':
        print("\n--- Mode: Training and Backtesting ---")
        train_dataset = TimeSeriesDataset(X_train_flat, y_train_flat, WINDOW)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        criterion = nn.BCELoss(); optimizer = optim.Adam(model.parameters(), lr=LR)

        print("Training...")
        best_val_loss = float("inf")
        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad(); out = model(xb); loss = criterion(out, yb)
                loss.backward(); optimizer.step(); total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            model.eval()
            total_val_loss = 0; correct = 0; total = 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    out_test = model(xb); total_val_loss += criterion(out_test, yb).item()
                    preds = (out_test > 0.5).float()
                    total += yb.size(0); correct += (preds == yb).sum().item()
            avg_val_loss = total_val_loss / len(test_loader); acc = correct / total
            print(f"Epoch {epoch}/{EPOCHS}  TrainLoss: {avg_loss:.6f}  ValLoss: {avg_val_loss:.6f}  ValAcc: {acc:.4f}")
            if avg_val_loss < best_val_loss:
                print(f"  Validation loss improved. Saving best model to {BEST_MODEL_FILE}")
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), BEST_MODEL_FILE)
        
        print("\nTraining complete. Loading best model for backtest.")
        model.load_state_dict(torch.load(BEST_MODEL_FILE))

    else:
        print("\n--- Mode: Backtest-Only ---")
        print(f"Loading pre-trained model from {BEST_MODEL_FILE}...")
        try:
            model.load_state_dict(torch.load(BEST_MODEL_FILE))
        except FileNotFoundError:
            print(f"ERROR: Model file not found. Please run in 'train' mode first."); exit()
    
    print("\nRunning backtest...")
    balance = START_BALANCE; equity_curve = [balance]; position = 0.0
    entry_price = None; trade_log = []; cooldown_counter = 0

    all_probs = []
    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            all_probs.extend(model(xb.to(DEVICE)).cpu().numpy().flatten())

    start_idx_in_df = split_idx + WINDOW

    for t in range(len(all_probs)):
        if cooldown_counter > 0:
            cooldown_counter -= 1; equity_curve.append(balance); continue
        prob = float(all_probs[t]); pred = 1 if prob > 0.5 else 0
        df_idx = start_idx_in_df + t
        if df_idx >= len(df): break
        next_price_open = df['close_bonk'].iloc[df_idx]
        if balance < START_BALANCE * (1 - MAX_DRAWDOWN_PCT):
            print("Max drawdown hit, stopping trading."); break
        if pred == 1 and balance > 0 and position == 0:
            entry_slippage = random.uniform(SLIPPAGE_MIN, SLIPPAGE_MAX)
            exit_slippage = random.uniform(SLIPPAGE_MIN, SLIPPAGE_MAX)
            usd_to_use = balance * MAX_RISK_PCT; qty = usd_to_use / next_price_open
            entry_price = next_price_open * (1 + entry_slippage); position = qty
            fee = usd_to_use * TAKER_FEE_PCT; balance -= usd_to_use + fee; entry_usd = usd_to_use
            trade = {"entry_idx": df_idx, "entry_time": df['timestamp'].iloc[df_idx],
                     "entry_price": entry_price, "size_usd": usd_to_use, "qty": qty, "status": "open"}
            closed = False
            for h in range(1, MAX_HOLD_PERIOD + 1):
                check_idx = df_idx + h
                if check_idx >= len(df): break
                high, low = df['high_bonk'].iloc[check_idx], df['low_bonk'].iloc[check_idx]
                tp_price = entry_price * (1 + TAKE_PROFIT_PCT); sl_price = entry_price * (1 + STOP_LOSS_PCT)
                if low <= sl_price:
                    exit_price = sl_price * (1 - exit_slippage); exit_idx = check_idx
                    closed = True; reason = "stop_loss"
                elif high >= tp_price:
                    exit_price = tp_price * (1 - exit_slippage); exit_idx = check_idx
                    closed = True; reason = "take_profit"
                if closed:
                    exit_usd = position * exit_price; fee_exit = exit_usd * TAKER_FEE_PCT
                    balance += exit_usd - fee_exit; pnl = (exit_usd - fee_exit) - entry_usd - fee
                    trade.update({"exit_idx": exit_idx, "exit_time": df['timestamp'].iloc[exit_idx],
                                  "exit_price": exit_price, "pnl": pnl, "status": "closed", "reason": reason})
                    trade_log.append(trade); position = 0.0; entry_price = None
                    if pnl < 0: cooldown_counter = COOLDOWN_ON_LOSS
                    break
            if not closed:
                final_idx = min(df_idx + MAX_HOLD_PERIOD, len(df) - 1)
                exit_price = df['close_bonk'].iloc[final_idx] * (1 - exit_slippage)
                exit_usd = position * exit_price; fee_exit = exit_usd * TAKER_FEE_PCT
                balance += exit_usd - fee_exit; pnl = (exit_usd - fee_exit) - entry_usd - fee
                trade.update({"exit_idx": final_idx, "exit_time": df['timestamp'].iloc[final_idx],
                              "exit_price": exit_price, "pnl": pnl, "status": "closed", "reason": "timeout"})
                trade_log.append(trade); position = 0.0; entry_price = None
                if pnl < 0: cooldown_counter = COOLDOWN_ON_LOSS
        equity_curve.append(balance)
        
    trades_df = pd.DataFrame(trade_log)
    if not trades_df.empty:
        wins = trades_df[trades_df['pnl'] > 0].shape[0]; losses = trades_df[trades_df['pnl'] <= 0].shape[0]
        total_trades = len(trades_df); win_rate = wins / total_trades if total_trades > 0 else float('nan')
        print("\nBACKTEST SUMMARY")
        print(f"Start balance: ${START_BALANCE:.2f}"); print(f"End balance:   ${balance:.2f}")
        print(f"Total trades: {total_trades} | Wins: {wins} | Losses: {losses} | Win rate: {win_rate:.2%}")
        equity = pd.Series(equity_curve); periods_per_day = 24*60//5
        periods_per_year = periods_per_day*365
        period_returns = equity.pct_change().dropna().values
        ann_return = (equity.iloc[-1]/equity.iloc[0])**(periods_per_year/len(equity)) - 1 if len(equity) > 1 and equity.iloc[0] > 0 else 0
        sr = sharpe_ratio(period_returns, periods_per_year); mdd = max_drawdown(equity)
        print(f"Total return: {(equity.iloc[-1]/equity.iloc[0]-1):.4%}")
        print(f"Approx annualized return: {ann_return:.4%}"); print(f"Sharpe (approx): {sr:.4f}")
        print(f"Max drawdown: {mdd:.4%}")
        trades_df.to_csv(os.path.join(MODEL_PATH, "trade_log.csv"), index=False)
        pd.DataFrame({"equity": equity}).to_csv(os.path.join(MODEL_PATH, "equity_curve.csv"), index=False)
        plt.figure(figsize=(10,5)); plt.plot(equity.values); plt.title("Equity curve (realistic backtest)")
        plt.xlabel("Steps"); plt.ylabel("Balance (USD)"); plt.grid(True); plt.show()
    else:
        print("\nBACKTEST SUMMARY"); print(f"Start balance: ${START_BALANCE:.2f}")
        print(f"End balance:   ${balance:.2f}"); print("No trades were executed by the model.")