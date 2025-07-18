# entrenar_modelo_cierre_scdpx.py

import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from joblib import dump

# Archivos
DATASET_PATH = "data/dataset_entrenamiento.csv"
MODELO_PATH = "modelos/modelo_cierre_scdpx.pkl"
SCALER_PATH = "modelos/scaler_cierre_scdpx.pkl"

# Crear carpetas si no existen
os.makedirs("modelos", exist_ok=True)

# Cargar dataset
df = pd.read_csv(DATASET_PATH)

# Agregar una etiqueta binaria para cierre anticipado: si ganancia_futura < 0, considerar cierre
df["cerrar"] = (df["ganancia_futura"] < 0).astype(int)

# Variables de entrada
columnas_entrada = [
    'variacion', 'cuerpo', 'es_verde', 'fuerza_cuerpo', 'fuerza_mecha'
]

X = df[columnas_entrada]
y = df["cerrar"]

# Dividir y escalar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train_scaled, y_train)

# Evaluación
y_pred = modelo.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# Guardar modelo y scaler
dump(modelo, MODELO_PATH)
dump(scaler, SCALER_PATH)

print(f"✅ Modelo de cierre SCDP-X guardado exitosamente en: {MODELO_PATH}")

