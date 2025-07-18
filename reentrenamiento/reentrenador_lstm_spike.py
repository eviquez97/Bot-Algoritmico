import pandas as pd
import numpy as np
import os
import joblib
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config.paths import RUTA_DATA, RUTA_MODELOS
from utils.logs import log

def reentrenar_modelo_lstm_spike():
    try:
        ruta_dataset = os.path.join(RUTA_DATA, "dataset_spike_monstruo.csv")
        if not os.path.exists(ruta_dataset):
            log("⚠️ No se encontró el archivo dataset_spike_monstruo.csv")
            return

        df = pd.read_csv(ruta_dataset)
        df.columns = df.columns.str.strip()

        # Validación de columna spike
        if "spike" not in df.columns:
            log("⚠️ El dataset no contiene la columna 'spike'")
            return

        df = df[df["spike"].isin([0, 1])]

        if df.empty or len(df) < 50:
            log(f"⚠️ Dataset insuficiente para entrenar (actual: {len(df)} filas). Se requiere mínimo 50.")
            return

        columnas_entrada = ["open", "high", "low", "close", "spread", "momentum", "ema", "rsi"]
        X = df[columnas_entrada]
        y = df["spike"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = Sequential()
        model.add(LSTM(64, activation='relu', input_shape=(1, X_scaled.shape[2])))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        model.save(os.path.join(RUTA_MODELOS, "model_lstm_spike.h5"))
        joblib.dump(scaler, os.path.join(RUTA_MODELOS, "scaler_lstm_spike.pkl"))

        log("✅ Modelo LSTM Spike reentrenado correctamente.")

    except Exception as e:
        log(f"[❌ ERROR LSTM SPIKE] {str(e)}", nivel="error")

