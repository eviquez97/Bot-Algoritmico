# reentrenamiento/entrenador_drl.py

import numpy as np
import pandas as pd
import os
import joblib
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from utils.logs import log

def entrenar_modelo_drl(ruta_dataset="data/dataset_drl.csv",
                        ruta_modelo="modelos/model_drl.keras",
                        ruta_columnas="modelos/columnas_drl.pkl"):
    try:
        if not os.path.exists(ruta_dataset):
            log(f"[❌ DRL] Dataset no encontrado en {ruta_dataset}")
            return

        df = pd.read_csv(ruta_dataset)
        if len(df) < 30:
            log(f"[⛔ DRL] Dataset muy pequeño: solo {len(df)} filas. Mínimo requerido: 30.")
            return

        columnas_entrada = [
            "score", "futuro", "bajistas", "visual_spike", "rf_spike", "lstm_spike",
            "ema_diff", "rsi", "momentum", "spread", "monto", "multiplicador"
        ]
        columnas_salida = ["Q0", "Q1", "Q2", "Q3"]

        faltantes_in = [col for col in columnas_entrada if col not in df.columns]
        faltantes_out = [col for col in columnas_salida if col not in df.columns]

        if faltantes_in:
            log(f"[❌ DRL] Faltan columnas de entrada: {faltantes_in}")
            return
        if faltantes_out:
            log(f"[❌ DRL] Faltan columnas de salida: {faltantes_out}")
            return

        log(f"[📊 DRL] Dataset cargado correctamente: {df.shape[0]} filas")

        X = df[columnas_entrada].copy()
        X = X / np.maximum(X.max(), 1)  # Escalado simple
        y = df[columnas_salida].values

        X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42)

        model = Sequential()
        model.add(InputLayer(input_shape=(X.shape[1],)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(4, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

        log(f"[⚙️ DRL] Entrenando modelo... X: {X.shape}, y: {y.shape}")
        model.fit(X_train, y_train, epochs=20, verbose=0)

        model.save(ruta_modelo)
        joblib.dump(columnas_entrada, ruta_columnas)

        log(f"[✅ DRL] Modelo guardado en {ruta_modelo}")
        log(f"[📁 DRL] Columnas guardadas en {ruta_columnas}")

    except Exception as e:
        log(f"[❌ ERROR DRL ENTRENAMIENTO] {e}")

# Ejecución directa (opcional desde consola)
if __name__ == "__main__":
    entrenar_modelo_drl()

