import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# Rutas
RUTA_CSV = "data/dataset_spike_monstruo_limpio.csv"
RUTA_MODELO = "modelos/scs_vision_x_model.keras"
RUTA_SCALER = "modelos/scaler_visual_spike.pkl"

# Columnas esperadas
COLUMNAS = ['fuerza_cuerpo', 'fuerza_mecha', 'bajistas', 'rsi', 'momentum',
            'spread', 'score', 'ema', 'variacion']

print("🔍 Cargando dataset...")
df = pd.read_csv(RUTA_CSV)

if df.shape[0] < 60:
    print("❌ No hay suficientes filas para evaluar (mínimo 60 requeridas).")
    exit()

# Preprocesamiento
df = df.dropna()
df = df[COLUMNAS]

scaler = joblib.load(RUTA_SCALER)
modelo = load_model(RUTA_MODELO)

X = df.values
X_scaled = scaler.transform(X)
X_scaled = np.expand_dims(X_scaled, axis=1)  # [samples, 1, features]

# Obtener solo la última fila para la prueba de anticipación
X_final = X_scaled[-1:]

print("\n🧬 Input shape final:", X_final.shape)
print("📊 Última fila que se le pasa al modelo:\n", pd.DataFrame(df.tail(1).values, columns=COLUMNAS))

# Predicción
pred = modelo.predict(X_final)
pred_valor = float(pred[0][0])

print(f"\n🧠 Predicción modelo Visual CNN (spike anticipado): {pred_valor:.4f}")
