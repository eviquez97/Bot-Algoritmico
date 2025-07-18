# test_spike_modelos_v5.py

import pandas as pd
import numpy as np
import tensorflow as tf
from modelos.modelos_spike import model_spike, model_lstm_spike, scs_vision_x_model, scaler_rf_2

# Columnas requeridas
columnas_rf_lstm = list(scaler_rf_2.feature_names_in_)
columnas_visual = ['fuerza_cuerpo', 'fuerza_mecha', 'bajistas', 'rsi', 'momentum', 'spread', 'score', 'variacion']

# Cargar CSV
df = pd.read_csv("data/contexto_historico.csv")
df = df.dropna()

# Verificar filas suficientes
if len(df) < 60:
    print(f"[❌] No hay suficientes filas en el CSV. Solo hay {len(df)}")
    exit()

# Recorte y preparación
df = df.tail(60).copy()
df_rf_lstm = df[columnas_rf_lstm].tail(30).astype('float64')
df_visual = df[columnas_visual].tail(30).astype('float64')

x_rf = scaler_rf_2.transform(df_rf_lstm)
x_seq = np.reshape(x_rf, (1, 30, len(columnas_rf_lstm)))
x_vis = np.reshape(df_visual.values, (1, 30, 8))

# Predicciones
print("\n[🧪 TEST SPIKE IA MODELOS V5]")
try:
    pred_rf = float(model_spike.predict_proba(x_rf)[-1][1])
    print(f"🔵 RF Spike:     {pred_rf:.4f}")
except Exception as e:
    print(f"[❌ RF SPIKE ERROR] {e}")

try:
    pred_lstm = float(model_lstm_spike(x_seq, training=False).numpy().flatten()[0])
    print(f"🟣 LSTM Spike:   {pred_lstm:.4f}")
except Exception as e:
    print(f"[❌ LSTM SPIKE ERROR] {e}")

try:
    pred_visual = float(scs_vision_x_model(x_vis, training=False).numpy().flatten()[0])
    print(f"🟡 Visual Spike: {pred_visual:.4f}")
except Exception as e:
    print(f"[❌ VISUAL SPIKE ERROR] {e}")
