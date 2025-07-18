import pandas as pd
import numpy as np

# Cargar dataset original
archivo_entrada = "data/dataset_spike_monstruo.csv"
archivo_salida = "data/dataset_spike_monstruo_limpio.csv"

# Leer CSV
df = pd.read_csv(archivo_entrada)

# Eliminar columnas inútiles o corruptas
if "target" in df.columns:
    df = df.drop(columns=["target"])

# Verificar y calcular columnas faltantes
if "variacion" not in df.columns:
    df["variacion"] = (df["close"] - df["open"]) / df["open"]

if "score" not in df.columns:
    df["score"] = (df["close"] - df["ema"]) * df["momentum"]

# Eliminar filas con NaNs en columnas clave
columnas_clave = ["open", "high", "low", "close", "spread", "ema", "rsi", "momentum", "score", "variacion"]
df = df.dropna(subset=columnas_clave)

# Reordenar columnas si es necesario
columnas_ordenadas = ["epoch", "open", "high", "low", "close", "spread", "ema", "rsi", "momentum", "score", "variacion"]
df = df[columnas_ordenadas]

# Guardar CSV limpio
df.to_csv(archivo_salida, index=False)
print(f"✅ Dataset limpio guardado en: {archivo_salida}")
