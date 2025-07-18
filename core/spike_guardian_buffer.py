import pandas as pd
import numpy as np
from utils.log import log
from tensorflow.keras.models import load_model
import joblib

# === CARGA DE MODELOS IA DE SPIKE ===
scaler_rf = joblib.load("modelos/scaler_rf_2.pkl")
model_rf = joblib.load("modelos/model_spike.pkl")
model_lstm = load_model("modelos/model_lstm_spike.h5")

# === FUNCIÓN PRINCIPAL DE EVALUACIÓN SPIKE ===
def evaluar_spike_anticipado(df_contexto):
    try:
        columnas = ["open", "high", "low", "close", "spread", "momentum", "ema", "rsi"]
        df_filtrado = df_contexto[columnas].copy()

        # RandomForest
        X_rf = scaler_rf.transform(df_filtrado.tail(30))
        pred_rf = model_rf.predict(X_rf)[-1]

        # LSTM
        X_lstm = scaler_rf.transform(df_filtrado.tail(60))
        X_lstm = X_lstm.reshape(1, 60, 8)
        pred_lstm = model_lstm.predict(X_lstm, verbose=0)[0][0]

        log(f"[🧠 SPIKE IA] RF: {pred_rf:.2f} | LSTM: {pred_lstm:.2f}")

        # Bloqueo si IA anticipa spike
        if pred_rf == 1.0 or pred_lstm > 0.5:
            log("🛡️ BLOQUEO ACTIVADO: Spike anticipado por IA")
            return True

        return False
    except Exception as e:
        log(f"[❌ ERROR SPIKE IA] {e}", "error")
        return False
