# entrenar_modelo_futuro.py

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RUTA_CSV = "data/contexto_historico.csv"
RUTA_SCALER = "modelos/scaler_futuro.pkl"
RUTA_MODELO = "modelos/modelo_prediccion_futura.pkl"

# Cargar dataset
df = pd.read_csv(RUTA_CSV)

# Validar columnas necesarias
columnas_requeridas = ['open', 'high', 'low', 'close', 'ema', 'rsi', 'momentum', 'futuro']
if not all(col in df.columns for col in columnas_requeridas):
    raise Exception(f"❌ Faltan columnas necesarias. Se esperaban: {columnas_requeridas}")

# Eliminar filas con nulos o infinitos
df = df.dropna()
df = df[~df.isin([float('inf'), float('-inf')]).any(axis=1)]

# Separar features y label
X = df[['open', 'high', 'low', 'close', 'ema', 'rsi', 'momentum']]
y = df['futuro']

# Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entrenar modelo
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Guardar scaler y modelo
joblib.dump(scaler, RUTA_SCALER)
joblib.dump(modelo, RUTA_MODELO)

print("✅ Modelo de predicción futura entrenado y guardado correctamente.")
