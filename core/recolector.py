import pandas as pd
from datetime import datetime
import os
from utils.logs import log

# 🔁 Buffer compartido para recolección
VELAS_RECOLECTADAS = []
RUTA_CSV = "dataset/dataset_operativo.csv"

# 📊 Indicadores técnicos
def calcular_indicadores(df):
    df = df.copy()
    df['ema'] = df['close'].ewm(span=10, adjust=False).mean()

    delta = df['close'].diff()
    ganancia = delta.clip(lower=0)
    perdida = -delta.clip(upper=0)
    avg_ganancia = ganancia.rolling(window=14).mean()
    avg_perdida = perdida.rolling(window=14).mean()
    rs = avg_ganancia / avg_perdida.replace(0, 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))

    df['momentum'] = df['close'] - df['close'].shift(4)
    df['spread'] = df['high'] - df['low']
    return df

# ⚡ Detección de spike en Boom 1000
def detectar_spike_boom1000(vela):
    try:
        return 1.0 if float(vela['close']) > float(vela['open']) else 0.0
    except Exception as e:
        log(f"[❌ ERROR DETECTOR SPIKE] {e}")
        return 0.0

# 💾 Guardado de datos enriquecidos
def guardar_csv_spike(df):
    os.makedirs("data", exist_ok=True)
    archivo_existe = os.path.exists(RUTA_CSV)
    columnas = ['open', 'high', 'low', 'close', 'spread', 'momentum', 'ema', 'rsi', 'spike']
    df[columnas].to_csv(RUTA_CSV, mode='a', header=not archivo_existe, index=False)

# 🔍 Procesamiento incremental de velas para spike
def procesar_vela_spike(vela):
    VELAS_RECOLECTADAS.append(vela)

    if len(VELAS_RECOLECTADAS) < 20:
        return  # No mostrar mensajes si aún es inicial

    df = pd.DataFrame(VELAS_RECOLECTADAS[-60:])
    df = calcular_indicadores(df)
    ultima = df.iloc[[-1]].copy()  # usar doble corchete para mantener DataFrame

    indicadores = ['spread', 'ema', 'rsi', 'momentum']
    if ultima[indicadores].isnull().any().any():
        return  # No guardar si hay NaN

    spike = detectar_spike_boom1000(ultima.iloc[0])
    ultima.loc[:, 'spike'] = spike
    guardar_csv_spike(ultima)

