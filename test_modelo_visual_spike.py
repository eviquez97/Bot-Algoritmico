# test_modelo_visual_spike.py
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Configuración
RUTA_CSV = "data/dataset_spike_monstruo_limpio.csv"
RUTA_MODELO = "modelos/scs_vision_x_model.keras"
RUTA_SCALER = "modelos/scaler_visual_spike.pkl"
COLUMNAS = ['fuerza_cuerpo', 'fuerza_mecha', 'bajistas', 'rsi', 'momentum', 'spread', 'score', 'ema', 'variacion']
TIMESTEPS = 32

# Cargar modelo y scaler
modelo = load_model(RUTA_MODELO)
scaler = joblib.load(RUTA_SCALER)

# Cargar dataset y limpiar
df = pd.read_csv(RUTA_CSV)
df = df.dropna()
df = df[COLUMNAS].copy()
df[COLUMNAS] = scaler.transform(df[COLUMNAS])

# Crear secuencias
def crear_secuencias(df, pasos):
    X = []
    for i in range(len(df) - pasos):
        X.append(df.iloc[i:i+pasos].values)
    return np.array(X)

X = crear_secuencias(df, TIMESTEPS)

# Tomar las últimas 20 secuencias para validar
ultimos = X[-20:]

# Predecir
predicciones = modelo.predict(ultimos)
for i, p in enumerate(predicciones, 1):
    print(f"[🧠 VISUAL SPIKE] Secuencia {i} → Predicción: {round(float(p), 2)}")
