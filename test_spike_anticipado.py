import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.ensemble import RandomForestClassifier

from modelos.modelos_spike import model_spike, model_lstm_spike, scaler_rf_2, scs_vision_x_model

# Ruta del CSV y columnas
ruta_csv = "data/dataset_spike_monstruo_limpio.csv"
columnas = ['fuerza_cuerpo', 'fuerza_mecha', 'bajistas', 'rsi', 'momentum', 'spread', 'score', 'ema', 'variacion']

# Cargar dataset y preparar inputs
df = pd.read_csv(ruta_csv).dropna()
df = df.tail(40).copy()

X = df[columnas].astype("float64").tail(30).copy()
y = df["spike_anticipado"].astype("int").tail(1).values[0]  # etiqueta real esperada

# Escalar y preparar inputs
X_rf = scaler_rf_2.transform(X)
X_seq = np.reshape(X_rf, (1, 30, len(columnas)))
X_vis = np.reshape(X.values, (1, 30, len(columnas)))

# Predicciones
pred_rf = model_spike.predict_proba(X_rf)[-1][1]
pred_lstm = model_lstm_spike(X_seq, training=False).numpy().flatten()[0]
pred_visual = scs_vision_x_model(X_vis, training=False).numpy().flatten()[0]

# Mostrar resultados
print("\n🧠 TEST SPIKE IA ANTICIPADO")
print(f"👉 Etiqueta real esperada (spike_anticipado): {y}")
print(f"📈 RF:     {pred_rf:.4f}")
print(f"📊 LSTM:   {pred_lstm:.4f}")
print(f"🎥 Visual: {pred_visual:.4f}")

if y == 1 and (pred_rf >= 0.7 or pred_lstm >= 0.6 or pred_visual >= 0.5):
    print("✅ El modelo anticipó correctamente el spike.")
elif y == 0 and (pred_rf < 0.7 and pred_lstm < 0.6 and pred_visual < 0.5):
    print("✅ El modelo no detectó spike, coherente con la etiqueta.")
else:
    print("⚠️ Desalineación entre predicción y etiqueta. Puede que el modelo no esté bien entrenado.")
