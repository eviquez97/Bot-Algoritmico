import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

# Rutas
RUTA_ORIGEN = "data/velas_1m.csv"
RUTA_DESTINO = "data/contexto_historico.csv"
RUTA_MODELO = "modelos/modelo_ganancia_rf.pkl"

# Verifica que el archivo fuente exista
if not os.path.exists(RUTA_ORIGEN):
    raise FileNotFoundError(f"No se encuentra el archivo: {RUTA_ORIGEN}")

# Cargar las velas originales
df = pd.read_csv(RUTA_ORIGEN)

# Validar columnas esenciales
columnas_requeridas = ['open', 'high', 'low', 'close']
for col in columnas_requeridas:
    if col not in df.columns:
        raise Exception(f"Falta la columna requerida: {col}")

# Agregar columnas de contexto
df["cuerpo"] = abs(df["close"] - df["open"])
df["spread"] = df["high"] - df["low"]
df["ema"] = df["close"].ewm(span=10).mean()
df["ema_diff"] = df["close"] - df["ema"]
df["rsi"] = df["close"].diff().apply(lambda x: x if x > 0 else 0).rolling(window=14).mean()
df["momentum"] = df["close"] - df["close"].shift(4)

# Dummy valores IA y decisión
df["score"] = np.random.uniform(-1, 1, len(df))
df["bajistas"] = np.random.uniform(0, 1, len(df))
df["visual_spike"] = 0
df["rf_spike"] = 0
df["lstm_spike"] = 0
df["monto"] = 0.0
df["multiplicador"] = 0.0

# Agregamos columna de 'futuro' como dummy
df["futuro"] = 0.5

# Limpiar nulos
df = df.dropna()

# Guardar CSV final
df.to_csv(RUTA_DESTINO, index=False)
print(f"✅ Dataset completo generado: {RUTA_DESTINO}")

# Entrenamiento del modelo de ganancia
X = df[["score", "bajistas", "ema_diff", "rsi", "momentum", "spread"]]
y = df["futuro"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Evaluar
y_pred = modelo.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"🎯 Modelo entrenado | MSE: {mse:.4f}")

# Guardar modelo
joblib.dump(modelo, RUTA_MODELO)
print(f"💾 Modelo guardado en: {RUTA_MODELO}")
