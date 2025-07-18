# entrenar_modelo_direccion_rf.py

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils.logs import log

# Carga el dataset
df = pd.read_csv("data/dataset_entrenamiento.csv")

# Definí las columnas de entrada
columnas_entrada = [
    "variacion", "cuerpo", "es_verde",
    "fuerza_cuerpo", "fuerza_mecha", "spread",
    "upper_shadow", "lower_shadow"
]

# Verifica columnas
for col in columnas_entrada + ["target_direccion"]:
    if col not in df.columns:
        raise ValueError(f"[❌ ERROR] Falta columna requerida en el dataset: {col}")

X = df[columnas_entrada]
y = df["target_direccion"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Validación
y_pred = modelo.predict(X_test)
log("[📊 EVALUACIÓN DIRECCIÓN]")
log("\n" + classification_report(y_test, y_pred))

# Guarda modelo y columnas
joblib.dump(modelo, "modelos/modelo_direccion_rf.pkl")
joblib.dump(columnas_entrada, "modelos/columnas_modelo_direccion.pkl")
log("✅ Modelo de dirección entrenado y guardado correctamente.")

