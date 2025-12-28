import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import *
import numpy as np
import tensorflow as tf
import pickle
import json
import matplotlib.pyplot as plt

print("\n" + "="*70)
print("📊 ÉVALUATION")
print("="*70 + "\n")

model = tf.keras.models.load_model(MODELS_DIR / "lstm_final.keras")
test_data = np.load(PROCESSED_DIR / "sequences_test.npz")
X_test, y_test = test_data['X'], test_data['y']

with open(PROCESSED_DIR / "scaler_y.pkl", 'rb') as f:
    scaler_y = pickle.load(f)

y_pred_scaled = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
y_test_real = scaler_y.inverse_transform(y_test)
y_pred_real = scaler_y.inverse_transform(y_pred_scaled)

rmse = np.sqrt(np.mean((y_test_real - y_pred_real)**2))
mae = np.mean(np.abs(y_test_real - y_pred_real))
mape = np.mean(np.abs((y_test_real - y_pred_real) / (y_test_real + 1e-8))) * 100

print(f"RMSE: {rmse:.2f} MW")
print(f"MAE:  {mae:.2f} MW")
print(f"MAPE: {mape:.2f}%\n")

with open(RESULTS_DIR / "test_metrics.json", 'w') as f:
    json.dump({'rmse': rmse, 'mae': mae, 'mape': mape}, f, indent=2)

indices = np.random.choice(len(X_test), 5, replace=False)
fig, axes = plt.subplots(5, 1, figsize=(14, 10))
fig.suptitle('Prédictions vs Réalité', fontsize=16)

for i, idx in enumerate(indices):
    axes[i].plot(y_test_real[idx], 'b-', label='Réel', linewidth=2)
    axes[i].plot(y_pred_real[idx], 'r--', label='Prédit', linewidth=2)
    axes[i].set_ylabel('MW')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / "test_predictions.png", dpi=150)
print(f"✅ Graphique: {VIZ_DIR / 'test_predictions.png'}\n")
print("="*70)
