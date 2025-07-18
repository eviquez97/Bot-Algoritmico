# generar_dataset_entrenamiento.py

import pandas as pd
import numpy as np
import os
from utils.logs import log

CSV = "data/contexto_historico.csv"
SALIDA = "data/dataset_entrenamiento.csv"

def calcular_variables(df):
    df = df.copy()
    df["variacion"] = df["close"] - df["open"]
    df["cuerpo"] = abs(df["variacion"])
    df["es_verde"] = (df["close"] > df["open"]).astype(int)
    df["fuerza_cuerpo"] = df["cuerpo"] / (df["high"] - df["low"] + 1e-6)
    df["fuerza_mecha"] = (df["high"] - df["low"]) / df["cuerpo"].replace(0, np.nan)
    df["spread"] = df["high"] - df["low"]
    df["upper_shadow"] = df["high"] - df[["close", "open"]].max(axis=1)
    df["lower_shadow"] = df[["close", "open"]].min(axis=1) - df["low"]
    return df

def generar_dataset():
    if not os.path.exists(CSV):
        log("[❌ ERROR] No se encontró el archivo contexto_historico.csv")
        return

    df = pd.read_csv(CSV)
    df = calcular_variables(df)

    # Creamos targets: ganancia futura y dirección futura
    df["target_ganancia"] = df["close"].shift(-2) - df["close"]
    df["target_direccion"] = (df["target_ganancia"] > 0).astype(int)

    df = df.dropna().reset_index(drop=True)
    df.to_csv(SALIDA, index=False)
    log(f"✅ Dataset de entrenamiento generado en: {SALIDA}")

if __name__ == "__main__":
    generar_dataset()

