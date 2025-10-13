# FILE: diagnose.py (ENHANCED FOR EXPERIMENTATION)
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ----------------------------------------------------
# --- KEY PARAMETERS FOR EXPERIMENTATION ---
# ----------------------------------------------------
# Change these values to test different hypotheses.
LOOKAHEAD_K = 6         # How many 5m candles into the future to predict (6 = 30 mins). Try 3 or 12.
PROFIT_THRESHOLD = 0.002 # The minimum % return to be considered a "win" (0.002 = 0.2%). Try 0.003.
# ----------------------------------------------------

# --- Data Loading and Feature Engineering ---
print("Loading and preparing data...")
df_bonk = pd.read_csv("data/bonk_usdt_BINANCE_history.csv")
df_btc = pd.read_csv("data/btc_usdt_BINANCE_history.csv")
df_bonk['timestamp'] = pd.to_datetime(df_bonk['timestamp'])
df_btc['timestamp'] = pd.to_datetime(df_btc['timestamp'])
df_bonk.set_index('timestamp', inplace=True)
df_btc.set_index('timestamp', inplace=True)
df = pd.merge(df_bonk, df_btc, left_index=True, right_index=True, suffixes=('_bonk', '_btc'))
df = df.reset_index()

def rsi(series, length=14):
    delta = series.diff(); up = delta.clip(lower=0).rolling(length).mean(); down = -delta.clip(upper=0).rolling(length).mean()
    rs = up / (down + 1e-12); return 100 - (100 / (1 + rs))

# --- Base and Relational Features ---
df['return'] = df['close_bonk'].pct_change()
df['log_return'] = np.log(df['close_bonk']).diff()
df['ma_fast'] = df['close_bonk'].rolling(10).mean()
df['ma_slow'] = df['close_bonk'].rolling(30).mean()
df['ma_ratio'] = df['ma_fast'] / df['ma_slow']
df['volatility'] = df['return'].rolling(30).std()
df['vol_change'] = df['volume_bonk'].pct_change()
df['rsi_14'] = rsi(df['close_bonk'], 14)
df['btc_return'] = df['close_btc'].pct_change()
for i in range(1, 4): df[f'btc_return_lag_{i}'] = df['btc_return'].shift(i)
df['bonk_btc_ratio'] = df['close_bonk'] / df['close_btc']
df['bonk_btc_ratio_ema_20'] = df['bonk_btc_ratio'].ewm(span=20, adjust=False).mean()
df['bonk_vs_btc_momentum'] = df['bonk_btc_ratio'] / df['bonk_btc_ratio_ema_20']
df['btc_atr'] = (df['high_btc'] - df['low_btc']).rolling(14).mean()
df['market_vol_ema_50'] = df['btc_atr'].ewm(span=50, adjust=False).mean()
df['market_vol_expansion'] = df['btc_atr'] / (df['market_vol_ema_50'] + 1e-12)

# --- NEW: Rate of Change Features ---
df['momentum_accel'] = df['bonk_vs_btc_momentum'].diff(periods=3)
df['vol_accel'] = df['market_vol_expansion'].diff(periods=3)

# --- Define the Target and the FINAL list of features ---
df['future_return'] = df['close_bonk'].shift(-LOOKAHEAD_K) / df['close_bonk'] - 1
# --- NEW: Using the smarter profit threshold for the target ---
df['target'] = (df['future_return'] > PROFIT_THRESHOLD).astype(int)

FEATURES = [
    'return','log_return','ma_fast','ma_slow','ma_ratio','volatility','vol_change','rsi_14',
    'btc_return_lag_1', 'btc_return_lag_2', 'btc_return_lag_3',
    'bonk_vs_btc_momentum', 'market_vol_expansion',
    'momentum_accel', 'vol_accel' # Added new features to the list
]
df.replace([np.inf, -np.inf], np.nan, inplace=True); df.dropna(inplace=True)
df.fillna(0, inplace=True); df = df.reset_index(drop=True)

# --- Prepare data for LightGBM (No sequences needed) ---
X = df[FEATURES]
y = df['target']

# --- Diagnostic Output: Check for severe class imbalance ---
print(f"\nTarget distribution with K={LOOKAHEAD_K} and Threshold={PROFIT_THRESHOLD*100:.2f}%:")
print(y.value_counts(normalize=True))

# Use a standard time-series split
split_idx = int(len(X) * 0.7)
X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# --- Train the LightGBM Model ---
print("\nTraining LightGBM model to diagnose feature effectiveness...")
lgb_train = lgb.Dataset(X_train_scaled, y_train)
lgb_val = lgb.Dataset(X_val_scaled, y_val, reference=lgb_train)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'n_estimators': 1000,
    'verbose': -1,
    'n_jobs': -1, # Use all available CPU cores
    'seed': 42
}

model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_val],
    callbacks=[lgb.early_stopping(100, verbose=True)]
)

# --- DIAGNOSE THE RESULTS ---
print("\n--- Diagnostic Results ---")
preds = model.predict(X_val_scaled, num_iteration=model.best_iteration)
binary_preds = (preds > 0.5).astype(int)
accuracy = accuracy_score(y_val, binary_preds)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

if accuracy > 0.53:
    print("CONCLUSION: The features appear to have a predictive signal! The problem was likely with the LSTM's ability to find it.")
    print("NEXT STEP: Transfer this winning feature/target logic back to your main backtest.py script.")
else:
    print("CONCLUSION: The features are still not predictive enough. The model is correctly identifying that there is no strong signal to learn.")
    print("NEXT STEP: Continue experimenting with LOOKAHEAD_K, PROFIT_THRESHOLD, and new feature ideas.")