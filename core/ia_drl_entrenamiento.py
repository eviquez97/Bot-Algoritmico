# core/ia_drl_entrenamiento.py

import pandas as pd
import numpy as np
import os
import tensorflow as tf
from core.ia_modelos import modelo_drl
from utils.logs import log

RUTA_DATASET = "data/experiencias_drl.csv"

def entrenar_drl_en_vivo():
    try:
        if not os.path.exists(RUTA_DATASET):
            log("[⛔ DRL] Dataset no encontrado para entrenamiento.")
            return

        df = pd.read_csv(RUTA_DATASET)
        if len(df) < 30:
            log("[⚠️ DRL] Dataset con pocas filas, entrenamiento omitido.")
            return

        columnas = ['score', 'porcentaje_bajistas', 'pred_futuro', 'monto']
        if not all(c in df.columns for c in columnas):
            log(f"[❌ ERROR DRL] Faltan columnas requeridas: {columnas}")
            return

        X = df[columnas].values.astype(np.float32).reshape(-1, 1, 4)  # 👈 reshape para (1,4)
        y = df["ganancia"].astype(np.float32).values  # 👈 salida esperada: valor real

        modelo_drl.fit(X, y, epochs=5, verbose=0)
        modelo_drl.save("modelos/model_drl.keras")
        log("✅ Reentrenamiento DRL completado.")

    except Exception as e:
        log(f"[❌ ERROR ENTRENAMIENTO DRL] {e}")


def reentrenar_modelo_drl():
    log("[🔁 DRL] Reentrenamiento DRL solicitado (método alterno).")
    entrenar_drl_en_vivo()




