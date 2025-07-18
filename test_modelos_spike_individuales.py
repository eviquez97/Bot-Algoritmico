import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Cargar modelos
modelo_rf = joblib.load("modelos/model_spike.pkl")
modelo_lstm = load_model("modelos/model_lstm_spike.keras")
modelo_visual = load_model("modelos/scs_vision_x_model.keras")
scaler_rf = joblib.load("modelos/scaler_rf_2.pkl")

# Cargar CSV
df = pd.read_csv("data/dataset_spike_monstruo_limpio.csv")

# Columnas requeridas
columnas = ['fuerza_cuerpo', 'fuerza_mecha', 'bajistas', 'rsi',
            'momentum', 'spread', 'score', 'ema', 'variacion']

df = df[columnas].dropna()

# Último bloque de 30 velas
bloque = df.tail(30)
X_rf = scaler_rf.transform(bloque)

# ==== Random Forest ====
pred_rf = modelo_rf.predict(X_rf)[-1]

# ==== LSTM ====
X_lstm = X_rf.reshape(1, 30, 9)
pred_lstm = modelo_lstm.predict(X_lstm)[0][0]

# ==== Visual ====
X_vis = X_rf.reshape(1, 30, 3, 3, 1)
pred_visual = modelo_visual.predict(X_vis)[0][0]

# Mostrar resultados
print("\n[🧠 TEST MODELOS SPIKE - INDIVIDUAL]")
print(f"Random Forest: {round(pred_rf, 4)}")
print(f"LSTM:          {round(pred_lstm, 4)}")
print(f"Visual:        {round(pred_visual, 4)}")

