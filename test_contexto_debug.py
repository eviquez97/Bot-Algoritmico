# test_contexto_debug.py

import pandas as pd
from core.contexto import construir_contexto

CSV_PATH = "data/contexto_historico.csv"

def testear_contexto():
    print("🔍 Cargando archivo CSV...")
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"✅ CSV cargado: {len(df)} filas, columnas: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Error al cargar CSV: {e}")
        return

    print("\n🔬 Ejecutando construir_contexto con cantidad=60...")
    contexto = construir_contexto(df, cantidad=60)

    print("\n📋 DataFrame tras limpieza:")
    df_limpio = df.copy()
    columnas_requeridas = ["open", "high", "low", "close"]
    df_limpio = df_limpio.dropna(subset=columnas_requeridas)
    df_limpio = df_limpio.tail(90).reset_index(drop=True)

    df_limpio["cuerpo"] = abs(df_limpio["close"] - df_limpio["open"])
    df_limpio["variacion"] = (df_limpio["close"] - df_limpio["open"]) / df_limpio["open"]
    df_limpio["fuerza_cuerpo"] = df_limpio["cuerpo"] / (df_limpio["high"] - df_limpio["low"]).replace(0, 1)
    df_limpio["fuerza_mecha"] = ((df_limpio["high"] - df_limpio["low"]) - df_limpio["cuerpo"]) / (df_limpio["high"] - df_limpio["low"]).replace(0, 1)
    df_limpio["spread"] = df_limpio["high"] - df_limpio["low"]
    df_limpio["score"] = df_limpio["variacion"].rolling(window=5, min_periods=1).mean()
    df_limpio["rsi"] = df_limpio["close"].diff().rolling(window=14, min_periods=1).mean().fillna(50)
    df_limpio["ema"] = df_limpio["close"].ewm(span=10, adjust=False).mean()
    df_limpio["momentum"] = df_limpio["close"].diff()

    df_limpio = df_limpio.replace([float("inf"), float("-inf")], 0.0)
    df_limpio = df_limpio.dropna()

    print(f"🧹 Filas válidas tras limpieza completa: {len(df_limpio)}")

    if contexto:
        print("\n✅ Contexto construido correctamente:")
        print(contexto)
    else:
        print("\n❌ Contexto NO construido.")

if __name__ == "__main__":
    testear_contexto()
