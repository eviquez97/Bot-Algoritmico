# crear_modelo_drl_dummy.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten, Input
from tensorflow.keras.optimizers import Adam

# Crear modelo dummy
model = Sequential([
    Input(shape=(1, 10)),
    LSTM(32, return_sequences=True),
    LSTM(16),
    Dense(16, activation='relu'),
    Dense(7, activation='softmax')  # 7 acciones posibles
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy')
model.save("modelos/modelo_drl.keras")
print("✅ Modelo DRL dummy creado correctamente.")
