# reentrenamiento/entrenar_modelo_futuro.py

import pandas as pd
import numpy as np
import os
import joblib
import sys
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime

# Cargar logger con path relativo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logs import log

RUTA_CSV = "data/contexto_historico.csv"
MODELO_PATH = "modelos/model_lstm_futuro.keras"

try:
    df = pd.read_csv(RUTA_CSV)

    # Asegurar columnas necesarias
    features = ['spread', 'ema', 'rsi', 'momentum']
    target = 'futuro'

    for col in features + [target]:
        if col not in df.columns:
            raise ValueError(f"[❌ ERROR] La columna requerida '{col}' no existe en el CSV.")

    df = df[features + [target]].dropna()

    X = df[features].values
    y = df[target].values

    X = X.reshape((X.shape[0], 1, X.shape[1]))  # Para LSTM

    # Modelo LSTM simple
    model = models.Sequential()
    model.add(layers.Input(shape=(1, len(features))))
    model.add(layers.LSTM(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))  # Salida continua

    model.compile(optimizer='adam', loss='mse')

    model.fit(X, y, epochs=20, batch_size=16, verbose=1)

    model.save(MODELO_PATH)
    log(f"[✅ FUTURO ENTRENADO] Modelo guardado en {MODELO_PATH}")

except Exception as e:
    log(f"[❌ ERROR ENTRENAMIENTO FUTURO] {e}")

