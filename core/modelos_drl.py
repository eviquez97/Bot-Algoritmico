# core/modelos_drl.py

import tensorflow as tf
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer
from keras.optimizers import Adam

MODELO_DRL_PATH = "modelos/model_drl.keras"

@tf.function(reduce_retracing=True)
def predecir_drl(modelo, estado):
    return modelo(estado, training=False)

def crear_modelo_drl():
    model = Sequential()
    model.add(InputLayer(input_shape=(1, 12)))  # ✅ 12 features por vela
    model.add(LSTM(64, activation='relu', return_sequences=False))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))  # ✅ Valor esperado para esa acción
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

try:
    modelo_drl = load_model(MODELO_DRL_PATH, compile=True)
    print("✅ Modelo DRL cargado exitosamente.")
except Exception as e:
    print(f"[⚠️ MODELO DRL NO ENCONTRADO] Se crea nuevo. Error: {e}")
    modelo_drl = crear_modelo_drl()

