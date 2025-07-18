# entrenar_modelo_spike.py

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

# Rutas
CSV_PATH = "data/dataset_spike_monstruo_limpio.csv"
MODEL_PATH = "modelos/model_spike.pkl"
SCALER_PATH = "modelos/scaler_rf_2.pkl"

# 1. Cargar CSV
df = pd.read_csv(CSV_PATH)

# 2. Verificar columnas
columnas_requeridas = [
    'fuerza_cuerpo', 'fuerza_mecha', 'bajistas', 'rsi',
    'momentum', 'spread', 'score', 'ema', 'variacion', 'spike'
]
faltantes = [col for col in columnas_requeridas if col not in df.columns]
if faltantes:
    raise ValueError(f"[❌ ERROR] Faltan columnas requeridas: {faltantes}")

# 3. Preprocesamiento
df = df.dropna()
X = df.drop("spike", axis=1)
y = df["spike"]

# 4. Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Entrenamiento del modelo
modelo = RandomForestClassifier(n_estimators=200, random_state=42)
modelo.fit(X_train, y_train)

# 7. Evaluación
y_pred = modelo.predict(X_test)
print("[🔍 EVALUACIÓN]")
print(classification_report(y_test, y_pred, digits=4))

# 8. Guardado del modelo y scaler
os.makedirs("modelos", exist_ok=True)
joblib.dump(modelo, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print(f"[✅ MODELO GUARDADO] {MODEL_PATH}")
print(f"[✅ SCALER GUARDADO] {SCALER_PATH}")
