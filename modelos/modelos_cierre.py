import joblib
import tensorflow as tf
import os

# Ruta base
BASE_PATH = "modelos"

# ✅ Carga del modelo de cierre entrenado
modelo_cierre = tf.keras.models.load_model(os.path.join(BASE_PATH, "model_cierre.keras"))

# ✅ Carga del scaler correspondiente
scaler_cierre = joblib.load(os.path.join(BASE_PATH, "scaler_cierre.pkl"))
