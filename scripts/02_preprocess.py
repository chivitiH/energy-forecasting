import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

print("\n" + "="*70)
print("🔧 PREPROCESSING")
print("="*70 + "\n")

csv_file = RAW_DIR / "PJME_hourly.csv"
if not csv_file.exists():
    print("❌ Fichier PJME_hourly.csv introuvable !")
    print("Lance d'abord : python scripts/01_download.py\n")
    sys.exit(1)

df = pd.read_csv(csv_file)
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.sort_values('Datetime').reset_index(drop=True)

print(f"✅ Chargé : {len(df):,} points")
print(f"   {df['Datetime'].min()} → {df['Datetime'].max()}\n")

df.rename(columns={'PJME_MW': 'consumption'}, inplace=True)
df['hour'] = df['Datetime'].dt.hour
df['day_of_week'] = df['Datetime'].dt.dayofweek
df['month'] = df['Datetime'].dt.month
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['lag_24h'] = df['consumption'].shift(24)
df['lag_168h'] = df['consumption'].shift(168)
df['rolling_mean_24h'] = df['consumption'].rolling(24).mean()
df['rolling_std_24h'] = df['consumption'].rolling(24).std()
df = df.dropna().reset_index(drop=True)

print(f"✅ Features créées : {df.shape[1]} colonnes")
print(f"   Après nettoyage : {len(df):,} points\n")

feature_cols = [c for c in FEATURES if c in df.columns]
data = df[feature_cols].values
target = df['consumption'].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()
data_scaled = scaler_X.fit_transform(data)
target_scaled = scaler_y.fit_transform(target.reshape(-1, 1)).flatten()

def create_sequences(data, target, lookback, horizon):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:i+lookback])
        y.append(target[i+lookback:i+lookback+horizon])
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, target_scaled, LOOKBACK, HORIZON)

print(f"✅ Séquences : X={X.shape} y={y.shape}\n")

n = len(X)
train_size = int(n * TRAIN_RATIO)
val_size = int(n * VAL_RATIO)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}\n")

np.savez_compressed(PROCESSED_DIR / "sequences_train.npz", X=X_train, y=y_train)
np.savez_compressed(PROCESSED_DIR / "sequences_val.npz", X=X_val, y=y_val)
np.savez_compressed(PROCESSED_DIR / "sequences_test.npz", X=X_test, y=y_test)

with open(PROCESSED_DIR / "scaler_X.pkl", 'wb') as f:
    pickle.dump(scaler_X, f)
with open(PROCESSED_DIR / "scaler_y.pkl", 'wb') as f:
    pickle.dump(scaler_y, f)

print("✅ Sauvegardé\n" + "="*70)
