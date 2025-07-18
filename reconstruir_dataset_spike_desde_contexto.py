import pandas as pd
import numpy as np
from utils.logs import log

ORIGEN = "data/contexto_historico.csv"
DESTINO = "data/dataset_spike_monstruo_limpio.csv"

try:
    df = pd.read_csv(ORIGEN)

    if not all(col in df.columns for col in ["open", "high", "low", "close"]):
        raise Exception("Faltan columnas base: open, high, low, close")

    df["spread"] = df["high"] - df["low"]
    df["momentum"] = df["close"].diff()
    df["variacion"] = (df["close"] - df["open"]) / df["open"]
    df["score"] = df["variacion"].rolling(window=5, min_periods=1).mean()
    df["rsi"] = 100 - (100 / (1 + (
        df["close"].diff().where(lambda x: x > 0, 0.0).rolling(window=14, min_periods=1).mean() /
        (-df["close"].diff().where(lambda x: x < 0, 0.0).rolling(window=14, min_periods=1).mean())
    )))
    df["ema"] = df["close"].ewm(span=10, adjust=False).mean()

    df["fuerza_cuerpo"] = abs(df["close"] - df["open"])
    df["mecha_superior"] = df["high"] - df[["close", "open"]].max(axis=1)
    df["mecha_inferior"] = df[["close", "open"]].min(axis=1) - df["low"]
    df["fuerza_mecha"] = df[["mecha_superior", "mecha_inferior"]].max(axis=1)
    df["bajistas"] = (df["close"] < df["open"]).rolling(window=5, min_periods=1).mean()

    if "spike" not in df.columns:
        df["spike"] = 0

    columnas_finales = [
        "fuerza_cuerpo", "fuerza_mecha", "bajistas",
        "rsi", "momentum", "spread", "score",
        "ema", "variacion", "spike"
    ]

    df = df[columnas_finales]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    df.to_csv(DESTINO, index=False)
    log(f"[✅ DATASET SPIKE RECONSTRUIDO] Dataset limpio desde contexto_historico.csv con {len(df)} filas.")

except Exception as e:
    log(f"[❌ ERROR AL RECONSTRUIR DESDE CONTEXTO] {e}")
