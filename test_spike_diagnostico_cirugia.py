import pandas as pd

CSV = "data/contexto_historico.csv"

# Paso 1: Cargar las últimas 120 velas crudas
df_crudo = pd.read_csv(CSV).tail(120)
print(f"\n[🔬 CIRUGÍA] Total filas cargadas: {len(df_crudo)}")

# Paso 2: Aplicar las transformaciones del bot
df = df_crudo.copy()
df["spread"] = df["high"] - df["low"]
df["momentum"] = df["close"].diff()
df["variacion"] = (df["close"] - df["open"]) / df["open"]
df["score"] = df["variacion"].rolling(window=5, min_periods=1).mean()
df["rsi"] = df["close"].diff().fillna(0)
df["ema"] = df["close"].ewm(span=10, adjust=False).mean()
df["fuerza_cuerpo"] = abs(df["close"] - df["open"])
df["mecha_superior"] = df["high"] - df[["close", "open"]].max(axis=1)
df["mecha_inferior"] = df[["close", "open"]].min(axis=1) - df["low"]
df["fuerza_mecha"] = df["mecha_superior"] + df["mecha_inferior"]
df["bajistas"] = (df["close"] < df["open"]).astype(int)

# Paso 3: Cortar a 60 filas post-transformación
df_60 = df.tail(60)
print(f"\n[📊 POST TRANSFORMACIÓN] 60 filas preparadas")

# Paso 4: Identificar columnas con NaNs
nulos_por_columna = df_60.isna().sum()
print("\n[📉 NULOS POR COLUMNA]")
print(nulos_por_columna[nulos_por_columna > 0])

# Paso 5: Mostrar las filas exactas con NaNs
filas_con_nan = df_60[df_60.isna().any(axis=1)]
print(f"\n[🚨 FILAS CON NULOS] {len(filas_con_nan)} encontradas")
print(filas_con_nan[["epoch", "open", "close", "rsi", "score", "ema"]].to_string(index=False))
