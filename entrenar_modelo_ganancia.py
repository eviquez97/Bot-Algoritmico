# entrenar_modelo_ganancia.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# Cargar el contexto
df = pd.read_csv("data/contexto_historico.csv")

# Validar columnas
requeridas = ['open', 'high', 'low', 'close', 'ema', 'rsi', 'momentum', 'ganancia_real']
faltantes = [c for c in requeridas if c not in df.columns]
if faltantes:
    raise Exception(f"❌ Columnas faltantes para entrenamiento de ganancia: {faltantes}")

# Preparar X y y
X = df[['open', 'high', 'low', 'close', 'ema', 'rsi', 'momentum']]
y = df['ganancia_real']

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entrenar
modelo = RandomForestRegressor(n_estimators=150, random_state=42)
modelo.fit(X_scaled, y)

# Crear carpeta si no existe
os.makedirs("modelos", exist_ok=True)

# Guardar modelo y scaler
joblib.dump(modelo, "modelos/modelo_rf_ganancia.pkl")
joblib.dump(scaler, "modelos/scaler_ganancia.pkl")

print("✅ Modelo de ganancia entrenado y guardado exitosamente.")
