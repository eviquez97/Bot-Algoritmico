# core/ia_ganancia.py

import joblib
import pandas as pd
from utils.logs import log

try:
    modelo = joblib.load("modelos/modelo_ganancia_rf.pkl")
    log("📈 Modelo de ganancia RF cargado correctamente.")
except Exception as e:
    log(f"[❌ ERROR GANANCIA RF] No se pudo cargar el modelo: {e}")
    modelo = None

COLUMNAS_REQUERIDAS = ['score', 'futuro', 'bajistas', 'rsi', 'momentum', 'spread']

def predecir_ganancia(contexto):
    if modelo is None:
        log("[⚠️ GANANCIA] Modelo no disponible.")
        return 0.0

    try:
        df = pd.DataFrame([contexto])

        for col in COLUMNAS_REQUERIDAS:
            if col not in df.columns:
                log(f"[⚠️ GANANCIA] Falta la columna '{col}' en el contexto.")
                return 0.0

        df = df[COLUMNAS_REQUERIDAS].replace([float("inf"), -float("inf")], 0).fillna(0)

        pred = modelo.predict(df)[0]
        return round(float(pred), 2)

    except Exception as e:
        log(f"[❌ ERROR GANANCIA PRED] {e}")
        return 0.0

def cargar_modelo_ganancia():
    return modelo

