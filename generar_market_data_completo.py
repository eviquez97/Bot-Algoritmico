import pandas as pd
import os

# Ruta de entrada y salida
ENTRADA = "data/market_data.csv"
SALIDA = "data/market_data.csv"

def calcular_indicadores(df):
    df["spread"] = df["high"] - df["low"]
    df["momentum"] = df["close"].diff()
    df["ema"] = df["close"].ewm(span=10, adjust=False).mean()
    df["rsi"] = calcular_rsi(df["close"])
    return df

def calcular_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def preparar_market_data():
    if not os.path.exists(ENTRADA):
        print(f"[❌ ERROR] No se encontró el archivo {ENTRADA}")
        return

    try:
        df = pd.read_csv(ENTRADA)
        print(f"[📥 CARGADO] {len(df)} filas cargadas desde {ENTRADA}")
        
        df = calcular_indicadores(df)
        df = df.dropna()

        if not os.path.exists("data"):
            os.makedirs("data")

        df.to_csv(SALIDA, index=False)
        print(f"[💾 GUARDADO] Archivo listo: {SALIDA} | Filas válidas: {len(df)}")

    except Exception as e:
        print(f"[❌ ERROR PROCESAMIENTO] {e}")

if __name__ == "__main__":
    preparar_market_data()
