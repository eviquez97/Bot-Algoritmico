import os
import joblib
import tensorflow as tf

# Rutas absolutas
MODELO_PATH = os.path.join("modelos", "model_lstm_futuro.keras")
SCALER_PATH = os.path.join("modelos", "scaler_futuro.pkl")

# Verificación de existencia
if not os.path.exists(MODELO_PATH):
    raise FileNotFoundError(f"[❌ ERROR] Modelo futuro no encontrado en {MODELO_PATH}")

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"[❌ ERROR] Scaler futuro no encontrado en {SCALER_PATH}")

# Carga del modelo LSTM
model_lstm_futuro = tf.keras.models.load_model(MODELO_PATH)

# Carga del scaler
scaler_futuro = joblib.load(SCALER_PATH)

# ✅ Función segura para predicción con LSTM
@tf.function(reduce_retracing=True)
def predecir_lstm_futuro(input_tensor):
    return model_lstm_futuro(input_tensor, training=False)

