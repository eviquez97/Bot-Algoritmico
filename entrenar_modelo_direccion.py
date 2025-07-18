import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# Ruta del CSV
RUTA = "data/dataset_drl.csv"

try:
    df = pd.read_csv(RUTA, on_bad_lines='skip')
except Exception as e:
    print(f"❌ Error al cargar el CSV: {e}")
    exit()

# Mostrar las columnas reales para ver cuál es el nombre correcto
print(f"🔍 Columnas detectadas en el dataset:\n{df.columns.tolist()}")

# Buscar una columna que contenga la palabra 'direccion'
columna_direccion = next((col for col in df.columns if "direccion" in col.lower()), None)

if not columna_direccion:
    print("❌ No se encontró una columna que contenga 'direccion'. Verifica el archivo.")
    exit()

print(f"✅ Usando columna '{columna_direccion}' como variable objetivo.")

# Filtrar datos válidos
df = df.dropna(subset=[columna_direccion])
df = df[df[columna_direccion].isin([0, 1])]

# Eliminar columnas objetivo y no predictoras
columnas_a_excluir = [columna_direccion, "ganancia_estimada", "duracion_estimada", "operacion_exitosa", "timestamp"]

columnas_a_excluir = [col for col in columnas_a_excluir if col in df.columns]
X = df.drop(columns=columnas_a_excluir)
y = df[columna_direccion]

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split y entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Guardar modelo
os.makedirs("modelos", exist_ok=True)
joblib.dump(modelo, "modelos/modelo_direccion_rf.pkl")
joblib.dump(scaler, "modelos/scaler_direccion_rf.pkl")

print("✅ Modelo de dirección entrenado y guardado correctamente.")

