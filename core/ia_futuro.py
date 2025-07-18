# core/ia_futuro.py

import joblib
import numpy as np
from utils.logs import log

# Carga los modelos y escalers
try:
    modelo_futuro = joblib.load("modelos/modelo_prediccion_futura.pkl")
    scaler_futuro = joblib.load("modelos/scaler_prediccion_futura.pkl")
    columnas_futuro = joblib.load("modelos/columnas_prediccion_futura.pkl")
    log("[✅ IA FUTURO] Modelos y columnas cargados correctamente.")
except Exception as e:
    modelo_futuro = None
    scaler_futuro = None
    columnas_futuro = None
    log(f"[❌ ERROR CARGA MODELOS FUTURO] {e}")

def predecir_futuro(df_contexto):
    try:
        if modelo_futuro is None or scaler_futuro is None or columnas_futuro is None:
            return None, None

        df = df_contexto.copy()

        # Asegura que estén todas las columnas necesarias
        for col in columnas_futuro:
            if col not in df.columns:
                raise ValueError(f"Columna faltante en el contexto: {col}")

        X = df[columnas_futuro].values
        X_scaled = scaler_futuro.transform(X)

        predicciones = modelo_futuro.predict(X_scaled)

        pred_futuro = float(predicciones[-1])
        ganancia_estim = round(pred_futuro * 100000, 2)

        return pred_futuro, ganancia_estim

    except Exception as e:
        log(f"[❌ ERROR PREDICCIÓN FUTURO] {e}")
        return None, None
