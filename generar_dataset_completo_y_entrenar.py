import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

# Paths
RUTA_ORIGEN = "data/velas_1m.csv"
RUTA_DESTINO = "data/contexto_historico.csv"
RUTA_MODELO = "modelos/modelo_ganancia_rf.pkl"

# Verificación
if not os.path.exists(RUTA_ORIGEN):
    raise FileNotFoundError(f"No se encuentra el archivo: {RUTA_ORIGEN}")

# Carga base
df = pd.read_csv(RUTA_ORIGEN).dropna()
if not {'open', 'high', 'low', 'close'}.issubset(df.columns):
    raise Exception("Faltan columnas OHLC en el archivo.")

# Cálculos de contexto
df["cuerpo"] = abs(df["close"] - df["open"])
df["spread"] = df["high"] - df["low"]
df["ema"] = df["close"].ewm(span=10).mean()
df["ema_diff"] = df["close"] - df["ema"]
df["rsi"] = 100 - (100 / (1 + df["close"].pct_change().rolling(14).mean()))
df["momentum"] = df["close"] - df["close"].shift(4)

# Etiqueta de ganancia futura
df["ganancia_futura"] = df["close"].shift(-5) - df["close"]

# Columnas dummy necesarias por el bot
df["score"] = np.random.uniform(-1, 1, len(df))
df["futuro"] = np.random.uniform(0, 1, len(df))
df["bajistas"] = np.random.uniform(0, 1, len(df))
df["visual_spike"] = 0.0
df["rf_spike"] = 0.0
df["lstm_spike"] = 0.0
df["monto"] = 0.0
df["multiplicador"] = 0.0

# Orden final y limpieza
columnas_finales = [
    "score", "futuro", "bajistas", "visual_spike", "rf_spike", "lstm_spike",
    "ema_diff", "rsi", "momentum", "spread", "monto", "multiplicador", "ganancia_futura"
]

df_final = df[columnas_finales].dropna()
df_final.to_csv(RUTA_DESTINO, index=False)
print(f"✅ Dataset completo generado: {RUTA_DESTINO}")

# Entrenamiento modelo
X = df_final.drop(columns=["ganancia_futura"])
y = df_final["ganancia_futura"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

joblib.dump(modelo, RUTA_MODELO)
print(f"🎯 Modelo entrenado | MSE: {mse:.4f}")
print(f"💾 Modelo guardado en: {RUTA_MODELO}")
