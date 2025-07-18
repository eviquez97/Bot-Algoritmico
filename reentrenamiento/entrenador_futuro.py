import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from utils.logs import log

DATASET_PATH = "data/contexto_historico.csv"
MODELO_PATH = "modelos/model_lstm_futuro.keras"
SCALER_PATH = "modelos/scaler_futuro.pkl"

def cargar_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError("No se encontró el dataset de contexto histórico.")

    df = pd.read_csv(DATASET_PATH)

    columnas = ["open", "high", "low", "close", "spread", "momentum", "ema", "rsi"]
    for col in columnas:
        if col not in df.columns:
            raise ValueError(f"Falta la columna requerida: {col}")

    df = df[columnas].replace([np.inf, -np.inf], np.nan).dropna()

    if df.shape[0] < 60:
        raise ValueError("No hay suficientes datos para entrenar el modelo (mínimo 60 filas).")

    return df

def preparar_datos(df):
    secuencia = 30
    X, y = [], []

    for i in range(len(df) - secuencia):
        ventana = df.iloc[i:i+secuencia]
        target = df.iloc[i+secuencia]["close"]
        y_actual = df.iloc[i+secuencia-1]["close"]

        if y_actual == 0:
            continue

        X.append(ventana.values)
        y.append(1 if target < y_actual else 0)  # 1 si bajará, 0 si subirá

    return np.array(X), np.array(y)

def escalar_datos(X):
    n_samples, n_steps, n_features = X.shape
    X_flat = X.reshape(-1, n_features)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat).reshape(n_samples, n_steps, n_features)

    return X_scaled, scaler

def construir_modelo(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def entrenar():
    try:
        log("[🔮 FUTURO] Preparando entrenamiento...")
        df = cargar_dataset()
        X, y = preparar_datos(df)
        X, scaler = escalar_datos(X)

        model = construir_modelo(input_shape=(X.shape[1], X.shape[2]))

        early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

        log("[📈 ENTRENAMIENTO] Iniciando entrenamiento del modelo futuro...")
        model.fit(X, y, epochs=20, batch_size=16, callbacks=[early_stop], verbose=0)

        os.makedirs("modelos", exist_ok=True)
        model.save(MODELO_PATH)
        joblib.dump(scaler, SCALER_PATH)

        log(f"[✅ FUTURO ENTRENADO] Modelo guardado en {MODELO_PATH}")
    except Exception as e:
        log(f"[❌ ERROR ENTRENAMIENTO FUTURO] {e}")

if __name__ == "__main__":
    entrenar()
