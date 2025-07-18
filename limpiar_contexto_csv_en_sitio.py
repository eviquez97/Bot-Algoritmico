import pandas as pd
from utils.logs import log

archivo = "data/contexto_historico.csv"

try:
    # Cargar forzando salto de líneas corruptas
    df = pd.read_csv(archivo, header=None, on_bad_lines='skip', engine='python')

    log(f"[🧹 LIMPIEZA EN SITIO] Cargadas {len(df)} filas antes de limpieza")

    # Asignar nombres esperados si no existen
    columnas = [
        "epoch", "open", "high", "low", "close", "spread", "momentum", "variacion",
        "score", "rsi", "ema", "fuerza_cuerpo", "mecha_superior", "mecha_inferior",
        "fuerza_mecha", "bajistas"
    ]
    df = df.iloc[:, :len(columnas)]
    df.columns = columnas

    # Eliminar filas con valores faltantes
    df = df.dropna()

    # Reescribir el mismo archivo limpio
    df.to_csv(archivo, index=False)
    log(f"[✅ LIMPIEZA COMPLETA] Archivo sobrescrito con {len(df)} filas válidas")

except Exception as e:
    log(f"[❌ ERROR LIMPIEZA EN SITIO] {e}")
