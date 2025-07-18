# reentrenamiento/autoentrenador.py

import os
import pandas as pd
from datetime import datetime, timedelta
from reentrenamiento.entrenador_drl import entrenar_modelo_drl
from reentrenamiento.entrenador_spike import entrenar_modelo_spike
from reentrenamiento.entrenador_cierre import entrenar_modelo_cierre
from utils.logs import log

RUTA_ULTIMA_EJECUCION = "data/ultima_ejecucion_reentrenamiento.txt"
DATASET_DRL = "data/dataset_drl.csv"
DATASET_SPIKE = "data/dataset_operativo.csv"
DATASET_CIERRE = "data/dataset_cierre.csv"

INTERVALO_HORAS = 3  # intervalo entre reentrenamientos automáticos

# Columnas obligatorias para validación
COLUMNAS_DRL = ['score', 'futuro', 'bajistas', 'visual_spike', 'rf_spike', 'lstm_spike',
                'ema_diff', 'rsi', 'momentum', 'spread', 'monto', 'multiplicador']
COLUMNAS_SPIKE = ['open', 'high', 'low', 'close', 'spread', 'momentum', 'ema', 'rsi', 'spike']
COLUMNAS_CIERRE = ['open', 'high', 'low', 'close', 'spread', 'momentum', 'ema', 'rsi', 'scm', 'scpx']

def dataset_valido(ruta, columnas_requeridas, minimo_filas=30):
    if not os.path.exists(ruta):
        return False, "[❌] No existe el archivo."
    try:
        df = pd.read_csv(ruta)
        if len(df) < minimo_filas:
            return False, f"[❌] Dataset con menos de {minimo_filas} filas ({len(df)})."
        if not all(col in df.columns for col in columnas_requeridas):
            return False, f"[❌] Faltan columnas requeridas: {columnas_requeridas}"
        return True, f"[✅] Dataset válido con {len(df)} filas."
    except Exception as e:
        return False, f"[❌] Error al leer CSV: {e}"

def verificar_reentrenamiento_general():
    try:
        ahora = datetime.now()

        # Verificar si existe registro previo
        if os.path.exists(RUTA_ULTIMA_EJECUCION):
            with open(RUTA_ULTIMA_EJECUCION, "r") as f:
                timestamp_str = f.read().strip()
                try:
                    ultima_ejecucion = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    log("[⚠️ FORMATO TIEMPO INVALIDO] Se forzará reentrenamiento por error de formato.")
                    ultima_ejecucion = datetime.min
                if ahora - ultima_ejecucion < timedelta(hours=INTERVALO_HORAS):
                    log("[🕒 AUTOENTRENAMIENTO] No ha pasado suficiente tiempo desde el último reentrenamiento.")
                    return

        log("♻️ [AUTOENTRENAMIENTO] Ejecutando reentrenamiento de modelos IA...")

        # DRL
        valido, msg = dataset_valido(DATASET_DRL, COLUMNAS_DRL)
        log(f"[🔍 DRL] {msg}")
        if valido:
            entrenar_modelo_drl()
        else:
            log("[⛔ DRL] Reentrenamiento omitido.")

        # SPIKE
        valido, msg = dataset_valido(DATASET_SPIKE, COLUMNAS_SPIKE)
        log(f"[🔍 SPIKE] {msg}")
        if valido:
            entrenar_modelo_spike()
        else:
            log("[⛔ SPIKE] Reentrenamiento omitido.")

        # CIERRE
        valido, msg = dataset_valido(DATASET_CIERRE, COLUMNAS_CIERRE)
        log(f"[🔍 CIERRE] {msg}")
        if valido:
            entrenar_modelo_cierre()
        else:
            log("[⛔ CIERRE] Reentrenamiento omitido.")

        # Guardar nueva marca de tiempo
        with open(RUTA_ULTIMA_EJECUCION, "w") as f:
            f.write(ahora.strftime("%Y-%m-%d %H:%M:%S"))

        log("[✅ AUTOENTRENAMIENTO COMPLETADO] Proceso finalizado.")

    except Exception as e:
        log(f"[❌ ERROR AUTOENTRENAMIENTO] {e}")

