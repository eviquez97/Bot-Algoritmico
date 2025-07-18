import pandas as pd
from utils.logs import log

RUTA_CSV = "data/dataset_spike_monstruo_limpio.csv"

COLUMNS_CORRECTAS = [
    "fuerza_cuerpo", "fuerza_mecha", "bajistas",
    "rsi", "momentum", "spread", "score",
    "ema", "variacion", "spike"
]

try:
    df = pd.read_csv(RUTA_CSV)
    log(f"[🔍 COLUMNAS ACTUALES] {list(df.columns)}")

    # Buscar columnas parecidas
    mapping = {}
    for col in COLUMNS_CORRECTAS:
        for actual in df.columns:
            if col in actual:
                mapping[actual] = col
                break

    df = df.rename(columns=mapping)

    # Forzar orden correcto y filtrar las columnas válidas
    df = df[COLUMNS_CORRECTAS]
    df = df.dropna()

    df.to_csv(RUTA_CSV, index=False)
    log(f"[✅ DATASET NORMALIZADO] Columnas forzadas y CSV sobrescrito.")

except Exception as e:
    log(f"[❌ ERROR FORZADO COLUMNAS] {e}")
