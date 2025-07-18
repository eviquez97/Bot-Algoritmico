import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import os

DATASET_PATH = "data/dataset_drl.csv"
MODELO_PATH = "modelos/model_drl.keras"

def entrenar_modelo_drl():
    if not os.path.exists(DATASET_PATH):
        print("[❌ ERROR] Dataset DRL no encontrado.")
        return

    df = pd.read_csv(DATASET_PATH)

    columnas_requeridas = [
        "score", "futuro", "bajistas",
        "visual_spike", "rf_spike", "lstm_spike",
        "ema_diff", "rsi", "momentum", "spread",
        "monto", "multiplicador"
    ]

    if not all(col in df.columns for col in columnas_requeridas):
        print("[❌ ERROR] Faltan columnas requeridas en el dataset DRL.")
        print(f"Se requieren: {columnas_requeridas}")
        return

    X = df[columnas_requeridas]
    y = df["score"]  # Se puede ajustar si tenés una columna de 'recompensa'

    if len(X) < 10:
        print(f"[⚠️ DATASET INSUFICIENTE] Solo hay {len(X)} muestras. Se requieren al menos 10 para entrenar.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        Input(shape=(X.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test), verbose=0)

    os.makedirs("modelos", exist_ok=True)
    model.save(MODELO_PATH)
    print("✅ Modelo DRL reentrenado y guardado como model_drl.keras")

if __name__ == "__main__":
    entrenar_modelo_drl()
