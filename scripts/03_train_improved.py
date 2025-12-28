"""Training AMÉLIORÉ - 3 LSTM + Attention sur données existantes"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import *
import numpy as np
import tensorflow as tf
import json
import time

print("\n" + "="*70)
print("⚡ TRAINING AMÉLIORÉ - 3 LSTM + ATTENTION")
print("="*70 + "\n")

print(f"Mixed Precision: {tf.keras.mixed_precision.global_policy().name}")
print(f"XLA: {tf.config.optimizer.get_jit()}")
print(f"GPUs: {len(tf.config.list_physical_devices('GPU'))}\n")

train_data = np.load(PROCESSED_DIR / "sequences_train.npz")
val_data = np.load(PROCESSED_DIR / "sequences_val.npz")
X_train, y_train = train_data['X'], train_data['y']
X_val, y_val = val_data['X'], val_data['y']

print(f"Train: {X_train.shape} | Val: {X_val.shape}\n")

def build_tf_dataset(X, y, batch_size, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(10000)
    ds = ds.batch(batch_size)
    if CACHE_TYPE == 'ram':
        ds = ds.cache()
    return ds.prefetch(AUTOTUNE)

train_ds = build_tf_dataset(X_train, y_train, BATCH_SIZE, True)
val_ds = build_tf_dataset(X_val, y_val, BATCH_SIZE, False)

n_features = X_train.shape[2]

print("🏗️  Construction modèle AMÉLIORÉ...\n")

inputs = tf.keras.layers.Input(shape=(LOOKBACK, n_features))

x = tf.keras.layers.LSTM(256, return_sequences=True)(inputs)
x = tf.keras.layers.Dropout(0.2)(x)

x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
x = tf.keras.layers.Dropout(0.2)(x)

x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
x = tf.keras.layers.Dropout(0.2)(x)

attention = tf.keras.layers.Dense(1, activation='tanh')(x)
attention = tf.keras.layers.Flatten()(attention)
attention = tf.keras.layers.Activation('softmax')(attention)
attention = tf.keras.layers.RepeatVector(64)(attention)
attention = tf.keras.layers.Permute([2, 1])(attention)

x_attended = tf.keras.layers.Multiply()([x, attention])
x_pooled = tf.keras.layers.Lambda(lambda xin: tf.keras.backend.sum(xin, axis=1))(x_attended)

x = tf.keras.layers.Dense(128, activation='relu')(x_pooled)
x = tf.keras.layers.Dropout(0.3)(x)

x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)

outputs = tf.keras.layers.Dense(HORIZON, dtype='float32')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
    loss='mse',
    metrics=['mae']
)

model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        str(CHECKPOINTS_DIR / "best_lstm_improved.keras"),
        save_best_only=True, monitor='val_loss', mode='min'
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6
    )
]

print("\n⏳ Training...\n")

start = time.time()

history = model.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS, callbacks=callbacks, verbose=1
)

duration = time.time() - start
print(f"\n⏱️ {duration:.1f}s ({duration/60:.1f} min)\n")

model.save(MODELS_DIR / "lstm_improved.keras")

metrics = {
    'best_val_loss': float(min(history.history['val_loss'])),
    'best_val_mae': float(min(history.history['val_mae'])),
    'training_time_seconds': duration,
    'epochs_completed': len(history.history['loss']),
    'architecture': '3_LSTM_Attention',
    'params': int(model.count_params())
}

with open(RESULTS_DIR / "metrics_improved.json", 'w') as f:
    json.dump(metrics, f, indent=2)

print("✅ Modèle amélioré sauvegardé")
print(f"✅ Paramètres: {model.count_params():,}")
print("\n📊 ARCHITECTURE :")
print("   V1 : 2 LSTM (128, 64)")
print("   V2 : 3 LSTM (256, 128, 64) + Attention\n")
print("="*70)
