import joblib
import tensorflow as tf
import os
from config import RUTA_MODELOS, log

def cargar_todos_los_modelos():
    modelos = {}

    try:
        modelos["scaler_rf_2"] = joblib.load(os.path.join(RUTA_MODELOS, "scaler_rf_2.pkl"))
        modelos["model_spike"] = joblib.load(os.path.join(RUTA_MODELOS, "model_spike.pkl"))
        modelos["model_lstm_spike"] = tf.keras.models.load_model(os.path.join(RUTA_MODELOS, "model_lstm_spike.keras"))
        modelos["scs_vision_x_model"] = tf.keras.models.load_model(os.path.join(RUTA_MODELOS, "scs_vision_x.keras"))
        modelos["model_lstm"] = tf.keras.models.load_model(os.path.join(RUTA_MODELOS, "model_lstm_v2.keras"))

        log("[✅ MODELOS IA CARGADOS] Todos los modelos se cargaron correctamente.")
    except Exception as e:
        log(f"[❌ ERROR CARGA MODELOS IA] {e}", "error")

    return modelos
