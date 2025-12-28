"""Configuration - Energy Forecasting"""
from pathlib import Path
import tensorflow as tf
import os

# ============================================================================
# OPTIMISATIONS GPU/CPU
# ============================================================================

tf.keras.mixed_precision.set_global_policy('mixed_float16')
tf.config.optimizer.set_jit(True)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
tf.config.experimental.enable_tensor_float_32_execution(True)
tf.config.run_functions_eagerly(False)
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '1'

AUTOTUNE = tf.data.AUTOTUNE

# ============================================================================
# CONSTANTES
# ============================================================================

PROJECT_NAME = "EnergyForecasting"
LOOKBACK = 168
HORIZON = 24
BATCH_SIZE = 512
EPOCHS = 100
LEARNING_RATE = 0.001
CACHE_TYPE = 'ram'

FEATURES = [
    'consumption', 'hour', 'day_of_week', 'month', 'is_weekend',
    'hour_sin', 'hour_cos', 'lag_24h', 'lag_168h',
    'rolling_mean_24h', 'rolling_std_24h'
]

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============================================================================
# CHEMINS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
VIZ_DIR = RESULTS_DIR / "visualizations"

def create_directories():
    for d in [RAW_DIR, PROCESSED_DIR, PREDICTIONS_DIR,
              CHECKPOINTS_DIR, RESULTS_DIR, VIZ_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"✅ Dossiers créés")

if __name__ == "__main__":
    create_directories()
    print(f"LOOKBACK: {LOOKBACK}h | HORIZON: {HORIZON}h | BATCH: {BATCH_SIZE}")
