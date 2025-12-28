"""Comparer V1 vs Improved"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import *
import numpy as np
import tensorflow as tf
import pickle
import json

print("\n" + "="*70)
print("📊 COMPARAISON V1 vs IMPROVED")
print("="*70 + "\n")

# Charger test set
test_data = np.load(PROCESSED_DIR / "sequences_test.npz")
X_test, y_test = test_data['X'], test_data['y']

with open(PROCESSED_DIR / "scaler_y.pkl", 'rb') as f:
    scaler_y = pickle.load(f)

# Charger modèles
print("📥 Chargement modèles...\n")
model_v1 = tf.keras.models.load_model(MODELS_DIR / "lstm_final.keras")
model_improved = tf.keras.models.load_model(MODELS_DIR / "lstm_improved.keras")

# Prédictions
print("🔮 Prédictions...\n")
y_pred_v1 = model_v1.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
y_pred_improved = model_improved.predict(X_test, batch_size=BATCH_SIZE, verbose=0)

# Dénormaliser
y_test_real = scaler_y.inverse_transform(y_test)
y_pred_v1_real = scaler_y.inverse_transform(y_pred_v1)
y_pred_improved_real = scaler_y.inverse_transform(y_pred_improved)

# Métriques
def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return rmse, mae, mape

rmse_v1, mae_v1, mape_v1 = calc_metrics(y_test_real, y_pred_v1_real)
rmse_improved, mae_improved, mape_improved = calc_metrics(y_test_real, y_pred_improved_real)

# Affichage
print("="*70)
print("📊 RÉSULTATS")
print("="*70 + "\n")

print("MODEL V1 (2 LSTM):")
print(f"  RMSE: {rmse_v1:.2f} MW")
print(f"  MAE:  {mae_v1:.2f} MW")
print(f"  MAPE: {mape_v1:.2f}%\n")

print("MODEL IMPROVED (3 LSTM + Attention):")
print(f"  RMSE: {rmse_improved:.2f} MW")
print(f"  MAE:  {mae_improved:.2f} MW")
print(f"  MAPE: {mape_improved:.2f}%\n")

# Amélioration
rmse_gain = ((rmse_v1 - rmse_improved) / rmse_v1) * 100
mae_gain = ((mae_v1 - mae_improved) / mae_v1) * 100
mape_gain = ((mape_v1 - mape_improved) / mape_v1) * 100

print("="*70)
print("📈 AMÉLIORATION")
print("="*70 + "\n")

if rmse_gain > 0:
    print(f"✅ RMSE: -{rmse_gain:.1f}% (meilleur)")
else:
    print(f"❌ RMSE: +{abs(rmse_gain):.1f}% (moins bon)")

if mae_gain > 0:
    print(f"✅ MAE:  -{mae_gain:.1f}% (meilleur)")
else:
    print(f"❌ MAE:  +{abs(mae_gain):.1f}% (moins bon)")

if mape_gain > 0:
    print(f"✅ MAPE: -{mape_gain:.1f}% (meilleur)")
else:
    print(f"❌ MAPE: +{abs(mape_gain):.1f}% (moins bon)")

print("\n" + "="*70)

# Sauvegarder
comparison = {
    'v1': {'rmse': float(rmse_v1), 'mae': float(mae_v1), 'mape': float(mape_v1)},
    'improved': {'rmse': float(rmse_improved), 'mae': float(mae_improved), 'mape': float(mape_improved)},
    'gain': {'rmse': float(rmse_gain), 'mae': float(mae_gain), 'mape': float(mape_gain)}
}

with open(RESULTS_DIR / "comparison.json", 'w') as f:
    json.dump(comparison, f, indent=2)

print("✅ Comparaison sauvegardée: results/comparison.json\n")

# Verdict
if mape_gain > 5:
    print("🏆 VERDICT: IMPROVED est SIGNIFICATIVEMENT meilleur ! ✅✅✅")
elif mape_gain > 0:
    print("✅ VERDICT: IMPROVED est légèrement meilleur")
elif mape_gain > -5:
    print("⚖️  VERDICT: Performance similaire (différence négligeable)")
else:
    print("❌ VERDICT: V1 était meilleur (overfitting du modèle improved ?)")

print("="*70)
