import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model

# Rutas de modelos y dataset
RUTA_CSV = "data/dataset_spike_monstruo_limpio.csv"
RUTA_SCALER = "modelos/scaler_spike.pkl"
RUTA_RF = "modelos/model_spike.pkl"
RUTA_LSTM = "modelos/model_lstm_spike.h5"
RUTA_CNN = "modelos/model_scs_vision_x.keras"

# Cargar dataset
df = pd.read_csv(RUTA_CSV)
df = df.dropna()

# Variables y objetivo
columnas = ['fuerza_cuerpo', 'fuerza_mecha', 'bajistas', 'rsi', 'momentum', 'spread', 'score', 'ema', 'variacion']
X = df[columnas].values
y = df['spike_anticipado'].values

# Escalar
with open(RUTA_SCALER, "rb") as f:
    scaler = pickle.load(f)

X_scaled = scaler.transform(X)

# ---------- RandomForest ----------
with open(RUTA_RF, "rb") as f:
    model_rf = pickle.load(f)

pred_rf = model_rf.predict(X_scaled)

# ---------- LSTM ----------
model_lstm = load_model(RUTA_LSTM)
X_lstm = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
pred_lstm = (model_lstm.predict(X_lstm, verbose=0) > 0.5).astype(int).flatten()

# ---------- CNN Visual ----------
model_cnn = load_model(RUTA_CNN)
X_cnn = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1, 1))
pred_cnn = (model_cnn.predict(X_cnn, verbose=0) > 0.5).astype(int).flatten()

# ---------- Evaluación ----------
spikes_reales = int(np.sum(y))
predichos_correctamente = int(np.sum((pred_rf + pred_lstm + pred_cnn >= 2) & (y == 1)))
precision_total = round(100 * predichos_correctamente / spikes_reales, 2) if spikes_reales > 0 else 0.0

print(f"\n✅ Total spikes anticipados reales: {spikes_reales}")
print(f"✅ Predichos correctamente: {predichos_correctamente}")
print(f"🎯 Precisión Spike IA anticipada: {precision_total}%")
