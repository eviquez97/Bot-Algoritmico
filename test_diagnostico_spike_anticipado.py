# test_diagnostico_spike_anticipado.py

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# Cargar CSV y limpiar
df = pd.read_csv("data/dataset_spike_monstruo_limpio.csv")
df = df.dropna().reset_index(drop=True)

# Últimas 120 velas
df_test = df.tail(120).copy()

# Definir columnas
columnas_rf_lstm = ['fuerza_cuerpo', 'fuerza_mecha', 'bajistas', 'rsi', 'momentum', 'spread', 'score', 'variacion']
columnas_visual = columnas_rf_lstm + ['ema']

# =========================
# 🔍 MODELO RF (SKLEARN)
# =========================
print("\n🔍 Cargando modelo RF...")
modelo_rf = joblib.load("modelos/model_spike.pkl")
scaler_rf = joblib.load("modelos/scaler_rf_2.pkl")

X_rf = scaler_rf.transform(df_test[columnas_rf_lstm])
pred_rf = modelo_rf.predict_proba(X_rf)[-1][1]  # Última fila
print(f"🧠 RF spike anticipado: {pred_rf:.4f}")

# =========================
# 🔍 MODELO LSTM
# =========================
print("\n🔍 Cargando modelo LSTM...")
modelo_lstm = load_model("modelos/model_lstm_spike.keras")

X_lstm_seq = []
for i in range(len(df_test) - 30, len(df_test)):
    secuencia = df_test[columnas_rf_lstm].iloc[i - 30:i]
    x_scaled = scaler_rf.transform(secuencia)
    X_lstm_seq.append(x_scaled)

X_lstm_seq = np.array(X_lstm_seq)
pred_lstm = modelo_lstm.predict(X_lstm_seq)[-1][0]
print(f"🧠 LSTM spike anticipado: {pred_lstm:.4f}")

# =========================
# 🔍 MODELO VISUAL
# =========================
print("\n🔍 Cargando modelo Visual...")
modelo_vis = load_model("modelos/scs_vision_x_model.keras")
scaler_vis = joblib.load("modelos/scaler_rf_2.pkl")  # Reutilizamos mismo scaler

X_vis = []
for i in range(len(df_test) - 30, len(df_test)):
    secuencia = df_test[columnas_visual].iloc[i - 30:i]
    x_scaled = scaler_vis.transform(secuencia)
    X_vis.append(x_scaled)

X_vis = np.array(X_vis)
pred_visual = modelo_vis.predict(X_vis)[-1][0]
print(f"🧠 Visual spike anticipado: {pred_visual:.4f}")

# =========================
# ✅ Diagnóstico final
# =========================
if sum([
    pred_rf >= 0.70,
    pred_lstm >= 0.60,
    pred_visual >= 0.50
]) >= 2:
    print("\n🛡️ CONSENSO: Spike anticipado detectado ✅")
else:
    print("\n🟢 No hay consenso de spike.")
