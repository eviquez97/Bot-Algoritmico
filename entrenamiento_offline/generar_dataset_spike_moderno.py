# generar_dataset_spike_moderno.py

import pandas as pd
import numpy as np
import os
from utils.logs import log

RUTA_ORIGEN = "data/contexto_historico.csv"
RUTA_SALIDA = "data/dataset_spike_monstruo_limpio.csv"

def calcular_columnas_modernas(df):
    df["fuerza_cuerpo"] = abs(df["close"] - df["open"])
    df["mecha_superior"] = df["high"] - df[["close", "open"]].max(axis=1)
    df["mecha_inferior"] = df[["close", "open"]].min(axis=1) - df["low"]
    df["fuerza_mecha"] = df["mecha_superior"] + df["mecha_inferior"]
    df["bajistas"] = np.where(df["close"] < df["open"], 1, 0)
    df["rsi"] = calcular_rsi(df["close"])
    df["momentum"] = df["close"].diff()
    df["spread"] = df["high"] - df["low"]
    df["score"] = df["spread"].rolling(window=5, min_periods=1).mean()
    return df

def calcular_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def etiquetar_spikes(df, umbral=2.5):
    df["target_spike"] = (df["fuerza_cuerpo"] > umbral).astype(int)
    return df

def limpiar_y_guardar():
    if not os.path.exists(RUTA_ORIGEN):
        log(f"[❌ ERROR] No se encontró el archivo original: {RUTA_ORIGEN}")
        return

    try:
        # Paso 1: Detectar columnas automáticamente
        with open(RUTA_ORIGEN, "r") as f:
            header = f.readline().strip().split(",")

        df = pd.read_csv(RUTA_ORIGEN, names=header, skiprows=1, on_bad_lines='skip')
        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df.tail(1000).copy()

        df = calcular_columnas_modernas(df)
        df = etiquetar_spikes(df)

        columnas_finales = [
            "fuerza_cuerpo", "fuerza_mecha", "bajistas", "rsi",
            "momentum", "spread", "score", "target_spike"
        ]
        df_final = df[columnas_finales].dropna()

        df_final.to_csv(RUTA_SALIDA, index=False)
        log(f"[✅ DATASET CREADO] {len(df_final)} filas guardadas en {RUTA_SALIDA}")
    except Exception as e:
        log(f"[❌ ERROR AL CREAR DATASET SPIKE] {e}")

if __name__ == "__main__":
    limpiar_y_guardar()
