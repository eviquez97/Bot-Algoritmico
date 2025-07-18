# corregir_contexto_csv.py

import pandas as pd
from utils.logs import log

RUTA_CSV = "data/contexto_historico.csv"
RUTA_SALIDA = "data/contexto_historico_limpio.csv"

COLUMNAS_VALIDAS = [
    "epoch", "open", "high", "low", "close",
    "spread", "momentum", "variacion", "score",
    "rsi", "ema"
]

try:
    df = pd.read_csv(RUTA_CSV, usecols=COLUMNAS_VALIDAS, on_bad_lines="skip")
    df = df.dropna()
    df.to_csv(RUTA_SALIDA, index=False)
    log(f"[✅ CSV LIMPIO] {len(df)} filas válidas guardadas en {RUTA_SALIDA}")
except Exception as e:
    log(f"[❌ ERROR AL REPARAR CSV] {e}")
