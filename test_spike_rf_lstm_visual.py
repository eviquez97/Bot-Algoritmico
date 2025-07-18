import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings("ignore")

from core.ia_spike import cargar_modelo_spike_rf, cargar_modelo_spike_lstm, cargar_modelo_spike_visual
from core.contexto import construir_contexto_para_spike
from utils.logs import log

# === Carga de modelos ===
modelo_rf = cargar_modelo_spike_rf()
modelo_lstm = cargar_modelo_spike_lstm()
modelo_visual = cargar_modelo_spike_visual()

# === Columnas específicas ===
columnas_rf = ['fuerza_cuerpo', 'fuerza_mecha', 'bajistas', 'rsi',
               'momentum', 'spread', 'score', 'variacion', 'ema']
columnas_lstm = ['fuerza_mecha', 'bajistas', 'rsi', 'momentum', 'spread', 'score', 'ema', 'variacion']
columnas_visual = ['fuerza_cuerpo', 'fuerza_mecha', 'bajistas', 'rsi',
                   'momentum', 'spread', 'score', 'variacion', 'ema']

# === Carga CSV y contexto ===
df = pd.read_csv("data/contexto_historico.csv")
df_contexto = construir_contexto_para_spike(df)

if df_contexto is None or len(df_contexto) < 32:
    print("❌ Contexto insuficiente para test.")
    exit()

print(f"[🧠 CONTEXTO SPIKE] Filas válidas tras limpieza: {len(df_contexto)}")

# === Test RF ===
X_rf = df_contexto[columnas_rf].tail(30)
scaler_rf = StandardScaler()
X_rf_scaled = scaler_rf.fit_transform(X_rf)
pred_rf = modelo_rf.predict_proba(X_rf_scaled)[-1][1]

# === Test LSTM ===
X_lstm = df_contexto[columnas_lstm].tail(30)
scaler_lstm = StandardScaler()
X_lstm_scaled = scaler_lstm.fit_transform(X_lstm)
X_lstm_input = np.expand_dims(X_lstm_scaled, axis=0)  # Shape: (1, 30, 8)
pred_lstm = modelo_lstm.predict(X_lstm_input, verbose=0)[0][0]

# === Test Visual ===
X_visual = df_contexto[columnas_visual].tail(32)
scaler_visual = StandardScaler()
X_visual_scaled = scaler_visual.fit_transform(X_visual)
X_visual_input = X_visual_scaled.reshape(1, 32, 9)
pred_visual = modelo_visual.predict(X_visual_input, verbose=0)[0][0]

# === Resultados ===
print(f"\n✅ Resultados del Test Spike IA:")
print(f"[🌲 RF SPIKE]:     {pred_rf:.4f}")
print(f"[🧠 LSTM SPIKE]:   {pred_lstm:.4f}")
print(f"[👁️ VISUAL SPIKE]: {pred_visual:.4f}")
