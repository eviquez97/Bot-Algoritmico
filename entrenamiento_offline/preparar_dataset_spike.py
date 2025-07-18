# preparar_dataset_spike.py

import pandas as pd

archivo_entrada = "data/dataset_spike_monstruo.csv"
archivo_salida = "data/dataset_spike_monstruo.csv"  # reemplaza sobre el mismo

try:
    df = pd.read_csv(archivo_entrada)

    columnas_requeridas = ["open", "close", "high", "low", "momentum", "spread", "score", "rsi", "ema"]
    df = df.dropna(subset=columnas_requeridas)

    # Generar etiqueta: si la vela es verde (spike), target = 1
    df["target"] = df.apply(lambda row: 1 if row["close"] > row["open"] else 0, axis=1)

    df.to_csv(archivo_salida, index=False)
    print(f"✅ Dataset preparado y guardado con columna 'target': {archivo_salida}")

except Exception as e:
    print(f"[❌ ERROR] No se pudo procesar: {e}")
