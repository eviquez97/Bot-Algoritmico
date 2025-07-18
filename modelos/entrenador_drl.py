import numpy as np
import pandas as pd
import os
import joblib
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

def reentrenar_drl(ruta_dataset="data/dataset_drl.csv", ruta_modelo="modelos/model_drl.keras", ruta_columnas="modelos/columnas_drl.pkl"):
    if not os.path.exists(ruta_dataset):
        print(f"[❌ ERROR] No se encontró el dataset en {ruta_dataset}")
        return

    df = pd.read_csv(ruta_dataset)

    columnas_entrada = [
        "score", "futuro", "bajistas", "visual_spike", "rf_spike", "lstm_spike",
        "ema_diff", "rsi", "momentum", "spread", "monto", "multiplicador"
    ]

    if not all(col in df.columns for col in columnas_entrada):
        print(f"[❌ ERROR] Dataset DRL incompleto. Faltan columnas requeridas.")
        print(f"Columnas encontradas: {df.columns.tolist()}")
        return

    # Entradas y normalización
    X = df[columnas_entrada].copy()
    X = X / np.maximum(X.max(), 1)  # Evita división por cero
    X_values = X.values

    # Salida: columnas Q0, Q1, Q2, Q3
    columnas_salida = ["Q0", "Q1", "Q2", "Q3"]
    if not all(col in df.columns for col in columnas_salida):
        print(f"[❌ ERROR] Faltan columnas de salida en el dataset: {columnas_salida}")
        return

    y = df[columnas_salida].values

    # División del dataset
    X_train, X_test, y_train, y_test = train_test_split(X_values, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(InputLayer(input_shape=(X_values.shape[1],)))  # Entrada plana de 12 features
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(4, activation='linear'))  # 4 Q-values
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    model.fit(X_train, y_train, epochs=20, verbose=0)

    # Guardado de modelo y columnas oficiales
    model.save(ruta_modelo)
    joblib.dump(columnas_entrada, ruta_columnas)

    print(f"✅ Modelo DRL reentrenado y guardado correctamente.")
    print(f"📊 Shape entrada: {X_values.shape} | Shape salida: {y.shape}")
    print(f"📁 Columnas entrada guardadas en: {ruta_columnas}")



