import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from tensorflow.keras.models import load_model

# === Cargar modelos y scaler ===
scaler_spike = joblib.load("modelos/scaler_rf_2.pkl")
model_rf_spike = joblib.load("modelos/model_spike.pkl")
model_lstm_spike = load_model("modelos/model_lstm_spike.keras")

# === Función principal de evaluación de spike anticipado ===
def evaluar_spike_anticipado(df_contexto):
    try:
        columnas_requeridas = ["open", "high", "low", "close", "spread", "momentum", "ema", "rsi"]
        df_filtrado = df_contexto[columnas_requeridas].copy()

        if df_filtrado.isnull().any().any():
            return False, 0.0, 0.0  # Valores faltantes, no evaluar

        # === Random Forest ===
        X_rf = scaler_spike.transform(df_filtrado)
        pred_rf = model_rf_spike.predict(X_rf)[-1]  # última predicción
        proba_rf = model_rf_spike.predict_proba(X_rf)[-1][1]  # prob spike

        # === LSTM ===
        X_lstm = X_rf.reshape(1, 30, 8)
        pred_lstm = model_lstm_spike.predict(X_lstm, verbose=0)[0][0]

        # === Decisión conjunta ===
        bloqueo = False
        if pred_rf == 1 and pred_lstm > 0.5:
            bloqueo = True

        # Imprimir en consola el resultado de las predicciones
        print(f"[📊 SPIKE PREDICTION] RF: {pred_rf}, Proba RF: {proba_rf:.2f}, LSTM: {pred_lstm:.2f}, Bloqueo: {bloqueo}")

        return bloqueo, proba_rf, pred_lstm

    except Exception as e:
        print(f"[❌ ERROR SPIKE IA] {e}")
        return False, 0.0, 0.0
