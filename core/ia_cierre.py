# core/ia_cierre.py

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os
from core.buffer import VELAS_BUFFER
from utils.logs import log
from tensorflow.keras.models import load_model

model_cierre = None
scaler_cierre = None

# ================================
# 🔁 Cargar modelo y scaler
# ================================
def cargar_modelo_cierre():
    global model_cierre, scaler_cierre
    try:
        model_path = "modelos/model_cierre.keras"
        scaler_path = "modelos/scaler_cierre.pkl"

        if not os.path.exists(model_path):
            log(f"[❌ SCDP-X] No se encontró el modelo en: {model_path}")
            return
        if not os.path.exists(scaler_path):
            log(f"[❌ SCDP-X] No se encontró el scaler en: {scaler_path}")
            return

        model_cierre = load_model(model_path, compile=False)
        scaler_cierre = joblib.load(scaler_path)
        log("[📥 SCDP-X] Modelo y scaler de cierre cargados correctamente.")
    except Exception as e:
        log(f"[❌ ERROR CARGA SCDP-X] {e}")
        model_cierre = None
        scaler_cierre = None

# Llamada automática al cargar el módulo
cargar_modelo_cierre()

# Alias para compatibilidad
cargar_modelo_scpx = cargar_modelo_cierre

def calcular_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

@tf.function(reduce_retracing=True)
def predecir_cierre(input_tensor):
    return model_cierre(input_tensor, training=False)

def construir_df_prediccion(velas):
    try:
        df = pd.DataFrame(velas[-60:])
        df["spread"] = df["high"] - df["low"]
        df["momentum"] = df["close"].diff()
        df["ema"] = df["close"].ewm(span=10, adjust=False).mean()
        df["rsi"] = calcular_rsi(df["close"])
        df["variacion"] = (df["close"] - df["open"]) / df["open"]
        df["score"] = df["variacion"].rolling(window=5, min_periods=1).mean()

        columnas = list(scaler_cierre.feature_names_in_)
        faltantes = [col for col in columnas if col not in df.columns]
        if faltantes:
            log(f"[❌ SCDP-X] Faltan columnas requeridas: {faltantes}")
            return None

        df = df[columnas].dropna().tail(30)
        if df.shape != (30, len(columnas)):
            log(f"[❌ SCDP-X] No hay suficientes filas tras limpieza: {df.shape[0]} / 30")
            return None

        X = scaler_cierre.transform(df)
        X = X.reshape((1, 30, len(columnas)))
        return X

    except Exception as e:
        log(f"[❌ ERROR construir_df_prediccion()] {e}")
        return None

def evaluar_scpx(df_actual):
    try:
        if model_cierre is None or scaler_cierre is None:
            log("[❌ SCDP-X] Modelo o scaler no cargados.")
            return False

        if len(VELAS_BUFFER) < 60:
            log("[⏳ SCDP-X] Aún no hay suficientes velas para evaluar cierre (min 60).")
            return False

        df_pred = construir_df_prediccion(VELAS_BUFFER)
        if df_pred is None:
            log("[⚠️ SCDP-X] No se pudo construir el DataFrame para predicción.")
            return False

        pred = predecir_cierre(df_pred).numpy()[0][0]
        if np.isnan(pred):
            log("[❌ SCDP-X] Predicción inválida (NaN)")
            return False

        log(f"[🔍 SCDP-X EVAL] Predicción cierre: {round(pred, 4)}")

        if pred > 0.6:
            log("✅ SCDP-X activa señal de cierre anticipado por IA.")
            return True

        ultima = df_actual.iloc[-1] if isinstance(df_actual, pd.DataFrame) else pd.Series(df_actual)
        spread = ultima.get("spread", 0)
        momentum = ultima.get("momentum", 0)
        color_verde = ultima.get("close", 0) > ultima.get("open", 0)

        if color_verde:
            log("🟢 [CIERRE SPIKE] Vela verde detectada. Se considera spike. Cierre forzado.")
            return True

        if spread > 1.0 and momentum < 0:
            log("[🛡️ DEFENSA] Cierre forzado por spread alto + momentum negativo.")
            return True

        return False

    except Exception as e:
        log(f"[❌ ERROR SCDP-X] {e}")
        return False
