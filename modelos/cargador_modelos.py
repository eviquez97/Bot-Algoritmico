import joblib
import tensorflow as tf

# CARGA DE MODELOS IA
scaler_rf_2 = joblib.load("modelos/scaler_rf_2.pkl")
model_spike = joblib.load("modelos/model_spike.pkl")
model_lstm_spike = tf.keras.models.load_model("modelos/model_lstm_spike.keras")
scs_vision_x_model = tf.keras.models.load_model("modelos/scs_vision_x_model.keras")
model_lstm = tf.keras.models.load_model("modelos/model_lstm_v2.keras")

# Modelos para cierre predictivo SCDP-X
from keras.models import load_model
import joblib

modelo_cierre = load_model("modelos/model_lstm_v2.keras")
scaler_cierre_predictivo = joblib.load("modelos/scaler_cierre_predictivo.pkl")


print("[✅ MODELOS IA CARGADOS] Todos los modelos se cargaron correctamente.")
