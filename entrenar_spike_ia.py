import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os

# Rutas
DATASET_PATH = "data/dataset_spike_monstruo_limpio.csv"
RUTA_RF = "modelos/model_spike.pkl"
RUTA_LSTM = "modelos/model_lstm_spike.keras"
RUTA_CNN = "modelos/scs_vision_x_model.keras"
RUTA_SCALER = "modelos/scaler_spike.pkl"

# Columnas usadas para los modelos
COLUMNAS = ['fuerza_cuerpo', 'fuerza_mecha', 'bajistas', 'rsi', 'momentum',
            'spread', 'score', 'ema', 'variacion']

# Cargar dataset
df = pd.read_csv(DATASET_PATH)

# Eliminar filas con NaNs
df = df.dropna(subset=COLUMNAS + ['spike_anticipado'])

# Separar variables y objetivo
X = df[COLUMNAS].copy()
y = df["spike_anticipado"].astype(int)

# Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------
# 1. Entrenar modelo RF
# --------------------------
rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
rf.fit(X_scaled, y)
joblib.dump(rf, RUTA_RF)
joblib.dump(scaler, RUTA_SCALER)
print("✅ RandomForest entrenado y guardado.")

# --------------------------
# 2. Entrenar modelo LSTM
# --------------------------
X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

lstm_model = Sequential([
    LSTM(32, input_shape=(1, X_scaled.shape[1]), return_sequences=False),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
lstm_model.fit(X_lstm, y, epochs=30, batch_size=16, verbose=0,
               callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

lstm_model.save(RUTA_LSTM)
print("✅ LSTM entrenado y guardado.")

# --------------------------
# 3. Entrenar modelo Visual CNN
# --------------------------
# Reshape para CNN: (samples, height, width, channels)
X_cnn = X_scaled.reshape((X_scaled.shape[0], 3, 3, 1))  # reshape 9 features en 3x3 imagen

cnn_model = Sequential([
    tf.keras.layers.Conv2D(16, (2,2), activation='relu', input_shape=(3,3,1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_cnn, y, epochs=30, batch_size=16, verbose=0,
              callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

cnn_model.save(RUTA_CNN)
print("✅ CNN Visual entrenado y guardado.")

print("🎯 ENTRENAMIENTO COMPLETADO - TODOS LOS MODELOS SPIKE IA GUARDADOS")
