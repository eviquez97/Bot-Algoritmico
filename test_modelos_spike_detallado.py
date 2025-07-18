# test_modelos_spike_detallado.py

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from modelos.modelos_spike import model_spike, model_lstm_spike, scs_vision_x_model, scaler_rf_2

# Cargar CSV original de entrenamiento (debes tenerlo con columnas limpias)
CSV = "data/dataset_spike_monstruo_limpio.csv"
df = pd.read_csv(CSV).dropna()

# Últimas 30 velas válidas
df_spike = df.tail(30).copy()

# Columnas usadas por el scaler (RandomForest + LSTM)
columnas = list(scaler_rf_2.feature_names_in_)
df_spike = df_spike[columnas].astype(float)

# Input para RF
x_rf = scaler_rf_2.transform(df_spike)

# Input para LSTM
x_lstm = np.reshape(x_rf, (1, 30, len(columnas)))

# Input para Visual (espera 30x8, no 3x3x1 si no fue entrenado así)
x_visual = np.reshape(x_rf, (1, 30, len(columnas)))  # Ajustar según cómo entrenaste

# Predicciones
print("\n[🧪 TEST PROFUNDO SPIKE MODELOS INDIVIDUALES]")
try:
    pred_rf = model_spike.predict_proba(x_rf)[-1][1]
    print(f"🔵 Random Forest: {pred_rf:.4f}")
except Exception as e:
    print(f"[❌ RF ERROR] {e}")

try:
    pred_lstm = model_lstm_spike(x_lstm, training=False).numpy().flatten()[0]
    print(f"🟣 LSTM: {pred_lstm:.4f}")
except Exception as e:
    print(f"[❌ LSTM ERROR] {e}")

try:
    pred_visual = scs_vision_x_model(x_visual, training=False).numpy().flatten()[0]
    print(f"🟢 Visual: {pred_visual:.4f}")
except Exception as e:
    print(f"[❌ VISUAL ERROR] {e}")
