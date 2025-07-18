import pandas as pd
import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
import joblib
from utils.logs import log

RUTA_DATASET = "data/dataset_spike_monstruo.csv"
RUTA_MODELO = "modelos/model_lstm_spike.h5"
RUTA_SCALER = "modelos/scaler_lstm_spike.pkl"

COLUMNAS = ["open", "high", "low", "close", "spread", "ema", "rsi", "momentum"]
TARGET = "spike"

def entrenar_lstm_spike():
    if not os.path.exists(RUTA_DATASET):
        log("[❌ ERROR] Dataset no encontrado.")
        return

    df = pd.read_csv(RUTA_DATASET).dropna()
    df = df[df["rsi"] > 0]

    if len(df) < 60:
        log("[⚠️ INSUFICIENTE] Dataset tiene menos de 60 filas válidas.")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[COLUMNAS])
    y = df[TARGET].values

    # Crear secuencias de 30 velas
    secuencias = []
    targets = []
    for i in range(30, len(X_scaled)):
        secuencias.append(X_scaled[i-30:i])
        targets.append(y[i])
    
    X = np.array(secuencias)
    y = np.array(targets)

    model = Sequential()
    model.add(LSTM(64, input_shape=(30, len(COLUMNAS)), return_sequences=False))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X, y, epochs=15, batch_size=16, verbose=1)

    os.makedirs("modelos", exist_ok=True)
    model.save(RUTA_MODELO)
    joblib.dump(scaler, RUTA_SCALER)

    log("[✅ ENTRENAMIENTO COMPLETO] Modelo LSTM Spike entrenado y guardado.")
    log(f"[💾 GUARDADO] Modelo en {RUTA_MODELO}")
    log(f"[💾 GUARDADO] Scaler en {RUTA_SCALER}")

if __name__ == "__main__":
    entrenar_lstm_spike()
