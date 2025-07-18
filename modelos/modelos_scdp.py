# modelos/modelos_scdp.py

import joblib
from keras.models import load_model

modelo_cierre = load_model("modelos/model_scdp.keras")
scaler_scdp = joblib.load("modelos/scaler_scdp.pkl")
