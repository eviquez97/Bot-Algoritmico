# core/verificaciones.py

import os
import pandas as pd
import subprocess
from utils.logs import log

def verificar_reentrenamiento_drl():
    try:
        ruta = "data/dataset_drl.csv"
        if not os.path.exists(ruta):
            return

        df = pd.read_csv(ruta)
        if df.shape[0] >= 10:
            log(f"[♻️ DRL AUTO] Reentrenando modelo con {df.shape[0]} muestras...")
            resultado = subprocess.run(["python", "reentrenamiento/entrenar_drl.py"], capture_output=True, text=True)
            if resultado.returncode == 0:
                log("[✅ DRL ENTRENADO] Modelo DRL actualizado automáticamente.")
            else:
                log(f"[❌ DRL ERROR ENTRENAMIENTO] {resultado.stderr}")
    except Exception as e:
        log(f"[❌ ERROR AUTO-DATASET] {e}")
