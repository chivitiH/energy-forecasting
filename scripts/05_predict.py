import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import *
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from datetime import datetime, timedelta

print("\n" + "="*70)
print("🔮 PRÉDICTIONS FUTURES")
print("="*70 + "\n")

model = tf.keras.models.load_model(MODELS_DIR / "lstm_final.keras")

with open(PROCESSED_DIR / "scaler_y.pkl", 'rb') as f:
    scaler_y = pickle.load(f)

test_data = np.load(PROCESSED_DIR / "sequences_test.npz")
last_sequence = test_data['X'][-1:, :, :]

y_pred_scaled = model.predict(last_sequence, verbose=0)
y_pred_real = scaler_y.inverse_transform(y_pred_scaled)

now = datetime.now()
future_times = [now + timedelta(hours=i) for i in range(1, HORIZON+1)]

forecast_df = pd.DataFrame({
    'datetime': future_times,
    'predicted_MW': y_pred_real[0]
})

print("Prédictions (premières 5h):")
print(forecast_df.head().to_string(index=False))
print(f"\n... ({HORIZON} heures totales)\n")

output = PREDICTIONS_DIR / f"forecast_{now.strftime('%Y%m%d_%H%M')}.csv"
forecast_df.to_csv(output, index=False)
print(f"✅ Sauvegardé: {output}\n")
print("="*70)
