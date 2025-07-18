# construir_dataset_spike_ia.py

import os
import csv
import time
from datetime import datetime
import pandas as pd
from core.buffer import VELAS_BUFFER

RUTA_CSV = "data/dataset_spike_moderno.csv"

ENCABEZADO = [
    "fuerza_cuerpo", "fuerza_mecha", "bajistas", "rsi", "momentum", "spread",
    "score", "ema", "variacion", "open", "close", "high", "low",
    "volumen_tick", "tiempo", "spike_real", "spike_anticipado"
]

def calcular_indicadores(df):
    df["spread"] = df["high"] - df["low"]
    df["variacion"] = df["close"].pct_change().fillna(0)
    df["ema"] = df["close"].ewm(span=10, adjust=False).mean()
    df["momentum"] = df["close"].diff().fillna(0)
    df["rsi"] = calcular_rsi(df["close"], periodos=14)
    df["score"] = (df["close"] - df["low"]) / df["spread"].replace(0, 0.0001)

    df["fuerza_cuerpo"] = ((df["close"] - df["open"]) > 0).astype(int)
    df["fuerza_mecha"] = ((df["high"] - df["close"]) > 0.2 * df["spread"]).astype(int)
    df["bajistas"] = (df["close"] < df["open"]).rolling(3).sum().fillna(0)

    df["volumen_tick"] = df.get("volumen_tick", 1)  # Dummy por ahora
    df["tiempo"] = datetime.now().isoformat()
    df["spike_real"] = (df["close"] > df["open"]).astype(int)
    df["spike_anticipado"] = 0  # Se marcarán después si aplica

    return df

def calcular_rsi(series, periodos=14):
    delta = series.diff()
    ganancia = delta.where(delta > 0, 0.0)
    perdida = -delta.where(delta < 0, 0.0)
    media_ganancia = ganancia.rolling(window=periodos).mean()
    media_perdida = perdida.rolling(window=periodos).mean()
    rs = media_ganancia / media_perdida.replace(0, 0.0001)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def guardar_vela(vela):
    if not os.path.exists(RUTA_CSV):
        with open(RUTA_CSV, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(ENCABEZADO)

    with open(RUTA_CSV, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([vela.get(col, "") for col in ENCABEZADO])

def actualizar_spike_anticipado(n=3):
    try:
        df = pd.read_csv(RUTA_CSV)
        if df.shape[0] < n + 1:
            return
        idx = df.shape[0] - 1
        df.loc[idx - n:idx - 1, "spike_anticipado"] = 1
        df.to_csv(RUTA_CSV, index=False)
        print(f"[🧠 ANTICIPADO] Marcadas {n} velas anteriores como spike_anticipado.")
    except Exception as e:
        print(f"[❌ ERROR ANTICIPADO] {e}")

def procesar_y_guardar(df_ultimas_120):
    if df_ultimas_120.shape[0] < 30:
        return

    try:
        df = df_ultimas_120.copy().reset_index(drop=True)
        df = calcular_indicadores(df)

        nueva_vela = df.iloc[-1]
        guardar_vela(nueva_vela)

        if nueva_vela["spike_real"] == 1:
            actualizar_spike_anticipado(3)

        print(f"[💾 NUEVA VELA] Spike real: {nueva_vela['spike_real']} - Vela guardada correctamente.")

    except Exception as e:
        print(f"[❌ ERROR GUARDADO DATASET SPIKE] {e}")

# === ⏱ LOOP DE MONITOREO EN TIEMPO REAL ===

print("[🚀 INICIO] Construcción del dataset spike activada...")

while True:
    try:
        if len(VELAS_BUFFER) >= 120:
            df_ultimas_120 = pd.DataFrame(VELAS_BUFFER[-120:])
            procesar_y_guardar(df_ultimas_120)
        time.sleep(2)  # Cada 2 segundos
    except KeyboardInterrupt:
        print("\n[⛔ PARADO MANUALMENTE]")
        break
    except Exception as e:
        print(f"[❌ ERROR GENERAL] {e}")
        time.sleep(2)
