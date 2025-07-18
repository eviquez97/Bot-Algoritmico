# entrenar_modelos_spike_todos.py

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras import Input

# === 1. CARGAR DATASET ORIGINAL ===
df = pd.read_csv("data/dataset_spike_monstruo_limpio.csv")  # Ojo: cambiar el nombre si usas otro

# === 2. FILTRAR COLUMNAS ===
columnas_features = ['fuerza_cuerpo', 'fuerza_mecha', 'bajistas', 'rsi', 'momentum', 'spread', 'score', 'ema', 'variacion']
X = df[columnas_features].copy()
y = df["spike"]

# === 3. ENTRENAR RANDOM FOREST (model_spike.pkl) ===
scaler_rf = StandardScaler()
X_scaled_rf = scaler_rf.fit_transform(X)

rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
rf.fit(X_scaled_rf, y)

# Guardar modelo y scaler
os.makedirs("modelos", exist_ok=True)
joblib.dump(rf, "modelos/model_spike.pkl")
joblib.dump(scaler_rf, "modelos/scaler_rf_2.pkl")

print("✅ RandomForest entrenado y guardado como 'model_spike.pkl'")

# === 4. ENTRENAR LSTM SECUENCIAL (model_lstm_spike.keras) ===
# Preparar secuencias para LSTM
SEQUENCE_LENGTH = 10
X_seq = []
y_seq = []

for i in range(len(X_scaled_rf) - SEQUENCE_LENGTH):
    X_seq.append(X_scaled_rf[i:i+SEQUENCE_LENGTH])
    y_seq.append(y.iloc[i+SEQUENCE_LENGTH])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

model_lstm = Sequential([
    LSTM(64, input_shape=(SEQUENCE_LENGTH, X.shape[1]), return_sequences=False),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model_lstm.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_lstm.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, validation_data=(X_test_lstm, y_test_lstm), callbacks=[es], verbose=0)

model_lstm.save("modelos/model_lstm_spike.keras")
print("✅ LSTM entrenado y guardado como 'model_lstm_spike.keras'")

# === 5. ENTRENAR MODELO VISUAL (scs_vision_x_model.keras) ===
# Simple red densa como placeholder visual
model_visual = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model_visual.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model_visual.fit(X_scaled_rf, y, epochs=30, batch_size=32, validation_split=0.2, verbose=0)

model_visual.save("modelos/scs_vision_x_model.keras")
print("✅ Modelo visual entrenado y guardado como 'scs_vision_x_model.keras'")
