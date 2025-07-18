# entrenar_modelo_lstm_futuro.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import joblib
import os

# 1. Cargar datos
df = pd.read_csv("./data/market_data.csv")

# 2. Validar columnas necesarias
columnas = ['open', 'high', 'low', 'close', 'spread', 'ema', 'rsi', 'momentum']
if not all(col in df.columns for col in columnas):
    raise ValueError(f"❌ El dataset debe contener las columnas: {columnas}")

# 3. Filtrar y escalar
df = df.dropna(subset=columnas)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[columnas])

# 4. Crear secuencias LSTM
X = []
y = []
timesteps = 30

for i in range(len(X_scaled) - timesteps):
    X.append(X_scaled[i:i+timesteps])
    y.append(1 if df["close"].iloc[i+timesteps] < df["close"].iloc[i+timesteps-1] else 0)

X = np.array(X)
y = np.array(y)

# 5. Definir y entrenar modelo
model = Sequential()
model.add(LSTM(50, return_sequences=False, input_shape=(timesteps, len(columnas))))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X, y, epochs=10, batch_size=32, verbose=1)

# 6. Guardar modelo y scaler
os.makedirs("modelos", exist_ok=True)
model.save("modelos/model_lstm_futuro.keras")
joblib.dump(scaler, "modelos/scaler_futuro.pkl")

print("✅ Modelo LSTM futuro y scaler guardados exitosamente.")
