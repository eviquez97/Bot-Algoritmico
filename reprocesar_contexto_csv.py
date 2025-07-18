import pandas as pd
import numpy as np
import os

CSV_ORIGEN = "data/contexto_historico.csv"
CSV_DESTINO = "data/contexto_historico.csv"  # Sobrescribe el original

if not os.path.exists(CSV_ORIGEN):
    print(f"❌ Archivo no encontrado: {CSV_ORIGEN}")
    exit()

df = pd.read_csv(CSV_ORIGEN)

if len(df) < 30:
    print("❌ No hay suficientes velas para procesar el contexto.")
    exit()

# Recalcular spread
df["spread"] = df["high"] - df["low"]

# EMA (exponential moving average)
df["ema"] = df["close"].ewm(span=10, adjust=False).mean()
df["ema_diff"] = df["close"] - df["ema"]

# RSI (Relative Strength Index)
delta = df["close"].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(window=14).mean()
avg_loss = pd.Series(loss).rolling(window=14).mean()
rs = avg_gain / (avg_loss + 1e-10)
df["rsi"] = 100 - (100 / (1 + rs))

# Momentum
df["momentum"] = df["close"] - df["close"].shift(4)

# Reordenar columnas (opcional)
columnas_finales = [
    'epoch', 'open', 'high', 'low', 'close',
    'spread', 'ema', 'ema_diff', 'rsi', 'momentum'
]
columnas_existentes = [col for col in columnas_finales if col in df.columns]
df = df[columnas_existentes + [col for col in df.columns if col not in columnas_existentes]]

# Guardar
df.to_csv(CSV_DESTINO, index=False)
print(f"✅ Contexto recalculado y guardado en: {CSV_DESTINO}")
