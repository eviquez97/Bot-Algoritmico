# entrenamiento_drl_ganancia_rf.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Ruta del CSV
csv_path = "data/contexto_historico.csv"

# Verificar existencia
if not os.path.exists(csv_path):
    print("❌ ERROR: El archivo contexto_historico.csv no existe.")
    exit()

# Cargar dataset
df = pd.read_csv(csv_path)

# Validar columnas necesarias
columnas = ['score', 'rsi', 'momentum', 'spread']
if not all(col in df.columns for col in columnas + ['futuro']):
    print(f"❌ ERROR: Faltan columnas requeridas: {columnas + ['futuro']}")
    exit()

# Filtrar y limpiar
df = df.dropna(subset=columnas + ['futuro'])

# Si no hay suficientes filas, salir
if len(df) < 120:
    print(f"⚠️ Aún no hay suficientes filas para reentrenar (actual: {len(df)}).")
    exit()

# Preparar datos
X = df[columnas]
y = df['futuro']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entrenar modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Guardar modelo y scaler
os.makedirs("modelos", exist_ok=True)
joblib.dump(model, "modelos/modelo_ganancia_rf.pkl")
joblib.dump(scaler, "modelos/scaler_ganancia_rf.pkl")
print("✅ Modelo de ganancia RF y scaler actualizados correctamente.")
