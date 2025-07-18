# modelos/modelos_spike.py

import joblib
from keras.models import load_model

# ✅ Modelos Spike IA
model_spike = joblib.load("modelos/model_spike.pkl")
model_lstm_spike = load_model("modelos/model_lstm_spike.h5")
scaler_rf_2 = joblib.load("modelos/scaler_rf_2.pkl")
scaler_visual = joblib.load("modelos/scaler_visual.pkl")  # ✅ Nuevo: scaler para modelo visual
scs_vision_x_model = load_model("modelos/scs_vision_x_model.keras")  # ✅ Renombrado correctamente

# ✅ Modelo de cierre
model_cierre = joblib.load("modelos/model_cierre.pkl")
