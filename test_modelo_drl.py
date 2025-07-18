# test_modelo_drl.py

import numpy as np
import joblib
from tensorflow.keras.models import load_model

print("🔍 Cargando modelo DRL y columnas...")
try:
    modelo = load_model("modelos/modelo_drl.keras")
    columnas = joblib.load("modelos/columnas_drl.pkl")
    print(f"✅ Modelo y columnas cargados correctamente. Columnas: {len(columnas)}")
except Exception as e:
    print(f"[❌ ERROR] No se pudo cargar el modelo o columnas: {e}")
    exit()

print("🧪 Generando datos sintéticos de entrada...")
X_dummy = np.random.rand(1, 60, len(columnas)).astype(np.float32)

print("📈 Ejecutando predicción...")
pred = modelo.predict(X_dummy, verbose=0)

print("📊 Resultado crudo del modelo:")
print(pred)
