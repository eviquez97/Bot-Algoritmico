# reentrenamiento/entrenador_spike.py

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from utils.logs import log

def entrenar_modelo_spike():
    try:
        ruta = "data/dataset_operativo.csv"

        if not os.path.exists(ruta):
            log("[❌ SPIKE] No se encontró el dataset.")
            return

        df = pd.read_csv(ruta)
        columnas_requeridas = ['open', 'high', 'low', 'close', 'spread', 'momentum', 'ema', 'rsi', 'spike']

        if not all(col in df.columns for col in columnas_requeridas):
            log(f"[❌ SPIKE] El dataset no contiene todas las columnas necesarias: {columnas_requeridas}")
            return

        df = df[columnas_requeridas].dropna()
        if len(df) < 30:
            log(f"[⛔ SPIKE] Dataset insuficiente: solo {len(df)} filas. Se requieren al menos 30.")
            return

        X = df.drop("spike", axis=1)
        y = df["spike"]

        # 🔄 Escalado
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 🌲 ENTRENAMIENTO RANDOM FOREST
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y)
        joblib.dump(rf, "modelos/model_spike.pkl")
        joblib.dump(scaler, "modelos/scaler_rf_2.pkl")
        log("[✅ SPIKE RF] Modelo Random Forest entrenado y guardado.")

        # 🧠 ENTRENAMIENTO LSTM
        X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        model_lstm = Sequential()
        model_lstm.add(LSTM(32, input_shape=(1, X_scaled.shape[1])))
        model_lstm.add(Dense(1, activation="sigmoid"))
        model_lstm.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])

        model_lstm.fit(X_lstm, y, epochs=10, batch_size=32, verbose=0)
        model_lstm.save("modelos/model_lstm_spike.h5")
        log("[✅ SPIKE LSTM] Modelo LSTM entrenado y guardado correctamente.")

    except Exception as e:
        log(f"[❌ ERROR SPIKE ENTRENAMIENTO] {e}")

# Exportación pública
__all__ = ["entrenar_modelo_spike"]





