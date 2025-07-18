import pandas as pd
from utils.logs import log
import os

DATASET_CIERRE = "data/dataset_cierre.csv"

def verificar_integridad_cierre():
    if not os.path.exists(DATASET_CIERRE):
        log("⚠️ Dataset de cierre no encontrado.")
        return

    try:
        df = pd.read_csv(DATASET_CIERRE)

        columnas_requeridas = ['resultado', 'open', 'high', 'low', 'close', 'spread', 'rsi', 'momentum']
        columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]

        if columnas_faltantes:
            log(f"❌ Faltan columnas requeridas: {columnas_faltantes}")
        else:
            log("✅ Dataset de cierre válido.")
    except Exception as e:
        log(f"❌ Error al leer o procesar el dataset de cierre: {e}")
