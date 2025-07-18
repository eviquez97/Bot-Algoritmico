# generar_futuro_y_entrenar.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("data/contexto_historico.csv")

# Verificamos columnas necesarias
columnas_requeridas = ['open', 'high', 'low', 'close', 'ema', 'rsi', 'momentum']
if not all(col in df.columns for col in columnas_requeridas + ['futuro']):
    raise Exception(f"❌ Faltan columnas requeridas: {columnas_requeridas + ['futuro']}")

# Quitamos la columna 'spread' si está
if "spread" in df.columns:
    df = df.drop(columns=["spread"])

X = df[columnas_requeridas]
y = df["futuro"]

# Entrenamiento
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_scaled, y_train)

# Guardar modelo y scaler
joblib.dump(modelo, "modelos/modelo_prediccion_futura.pkl")
joblib.dump(scaler, "modelos/escalador_futuro.pkl")

print("✅ Entrenamiento finalizado y modelos guardados.")

