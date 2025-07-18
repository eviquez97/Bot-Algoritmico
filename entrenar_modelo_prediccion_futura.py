# entrenar_modelo_prediccion_futura.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import dump
import os

# Rutas
DATASET_PATH = "data/dataset_entrenamiento.csv"
MODELO_PATH = "modelos/modelo_prediccion_futura.pkl"
SCALER_PATH = "modelos/scaler_prediccion_futura.pkl"

# Crear carpeta si no existe
os.makedirs("modelos", exist_ok=True)

# Cargar dataset
df = pd.read_csv(DATASET_PATH)

# Columnas candidatas para entrada
columnas_candidatas = [
    'open', 'high', 'low', 'close',
    'variacion', 'cuerpo', 'es_verde',
    'fuerza_cuerpo', 'fuerza_mecha', 'rsi',
    'volatilidad', 'tendencia',
    'fuerza_alcista', 'fuerza_bajista',
    'ema_10', 'ema_20',
    'momentum', 'macd', 'macd_signal'
]

# Filtrar solo las columnas que existen en el dataset actual
columnas_entrada = [col for col in columnas_candidatas if col in df.columns]
columna_objetivo = 'ganancia_futura'

if not columnas_entrada:
    raise Exception("❌ No se encontraron columnas válidas de entrada en el dataset.")

# Escalar datos
scaler = MinMaxScaler()
X = scaler.fit_transform(df[columnas_entrada])
y = df[columna_objetivo]

# División train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Evaluar
predicciones = modelo.predict(X_test)
mse = mean_squared_error(y_test, predicciones)
print(f"📊 MSE de validación: {mse:.6f}")

# Guardar modelo y scaler
dump(modelo, MODELO_PATH)
dump(scaler, SCALER_PATH)
print(f"✅ Modelo y scaler de predicción futura guardados exitosamente.")

# ... después de entrenar el modelo y guardarlo

# Guardar el modelo, scaler y columnas
dump(modelo, "modelos/modelo_prediccion_futura.pkl")
dump(scaler, "modelos/scaler_prediccion_futura.pkl")
dump(columnas_entrada, "modelos/columnas_prediccion_futura.pkl")  # ← ESTA ES LA LÍNEA NUEVA

print("✅ Modelo y scaler de predicción futura guardados exitosamente.")

