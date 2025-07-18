import joblib
from keras.models import load_model

# Ruta del modelo y scaler
RUTA_MODELO = "modelos/model_lstm_futuro.keras"
RUTA_SCALER = "modelos/scaler_futuro.pkl"

try:
    model_lstm_futuro = load_model(RUTA_MODELO)
    print("✅ Modelo LSTM de predicción futura cargado correctamente.")
except Exception as e:
    print(f"[❌ ERROR] No se pudo cargar el modelo LSTM futuro: {e}")
    model_lstm_futuro = None

try:
    scaler_futuro = joblib.load(RUTA_SCALER)
    print("✅ Scaler de predicción futura cargado correctamente.")
except Exception as e:
    print(f"[❌ ERROR] No se pudo cargar el scaler de predicción futura: {e}")
    scaler_futuro = None
