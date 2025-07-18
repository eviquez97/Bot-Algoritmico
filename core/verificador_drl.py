# core/verificador_drl.py

import os
import datetime
import pandas as pd
import numpy as np
from keras.models import load_model
from utils.logs import log

COLUMNAS_DRL = [
    "score", "ganancia_esperada", "prediccion_futura", "porcentaje_bajistas",
    "rsi_29", "momentum_29", "spread_29", "ema_29",
    "spike_rf", "spike_lstm", "spike_visual", "duracion_estim"
]

def verificar_integridad_drl():
    hoy = datetime.date.today()
    errores = []

    modelo_path = "modelos/model_drl.keras"
    dataset_path = "data/experiencias_drl.csv"

    # Verificar existencia
    if not os.path.exists(modelo_path):
        errores.append("❌ Falta el modelo model_drl.keras")
    else:
        fecha_modelo = datetime.date.fromtimestamp(os.path.getmtime(modelo_path))
        if fecha_modelo != hoy:
            errores.append("⚠️ El modelo DRL no fue actualizado hoy")

    if not os.path.exists(dataset_path):
        errores.append("❌ Falta el dataset experiencias_drl.csv")
    else:
        try:
            df = pd.read_csv(dataset_path)
            if df.shape[0] < 30:
                errores.append(f"⚠️ Dataset DRL con pocas filas: {df.shape[0]}")
        except Exception as e:
            errores.append(f"❌ Error al leer dataset DRL: {e}")

    # Validación del modelo en caliente con input correcto
    if os.path.exists(modelo_path):
        try:
            model = load_model(modelo_path)
            dummy_input = np.array([[1.0] * len(COLUMNAS_DRL)]).reshape(1, len(COLUMNAS_DRL))
            model.predict(dummy_input)
        except Exception as e:
            errores.append(f"❌ Error al usar model_drl.keras: {e}")

    # Log final
    if errores:
        log("[🧪 VERIFICACIÓN DRL] Fallos detectados:")
        for err in errores:
            log(err)
    else:
        log("✅ [VERIFICACIÓN DRL] Modelo y dataset en orden para operar.")

