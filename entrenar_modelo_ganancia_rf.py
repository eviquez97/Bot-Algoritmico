# entrenar_modelo_ganancia_rf.py

import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Cargar el dataset
CSV = "data/dataset_drl.csv"
df = pd.read_csv(CSV)

# Filtrar columnas válidas para features y la etiqueta 'Q3'
COLUMNAS_FEATURES = ["score", "rsi", "momentum", "spread"]
COLUMNA_OBJETIVO = "Q3"

# Limpiar y validar
df = df.dropna(subset=COLUMNAS_FEATURES + [COLUMNA_OBJETIVO])
X = df[COLUMNAS_FEATURES]
y = df[COLUMNA_OBJETIVO]

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entrenamiento
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_scaled, y)

# Guardar el modelo y el scaler
with open("modelos/modelo_ganancia_rf.pkl", "wb") as f:
    pickle.dump(modelo, f)

with open("modelos/scaler_ganancia_rf.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Modelo y scaler guardados correctamente.")
