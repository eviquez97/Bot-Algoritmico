import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv2D, MaxPooling2D, Flatten, InputLayer, Reshape
from keras.callbacks import EarlyStopping

# Cargar el dataset
df = pd.read_csv("data/dataset_spike_monstruo_limpio.csv")

# Validar columnas requeridas
columnas_requeridas = ['fuerza_cuerpo', 'fuerza_mecha', 'bajistas', 'rsi', 'momentum',
                       'spread', 'score', 'ema', 'variacion', 'spike_anticipado']
for col in columnas_requeridas:
    if col not in df.columns:
        raise ValueError(f"[❌ ERROR] Falta columna requerida: {col}")

df = df.dropna()

# -------------------- ENTRENAR MODELO RANDOM FOREST --------------------
print("🔁 Entrenando modelo RandomForest...")
X = df[columnas_requeridas[:-1]]
y = df['spike_anticipado']
model_rf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
model_rf.fit(X, y)
with open("modelos/model_spike.pkl", "wb") as f:
    pickle.dump(model_rf, f)
print("✅ Modelo RandomForest guardado como model_spike.pkl")

# -------------------- ENTRENAR MODELO LSTM --------------------
print("🔁 Preparando datos para LSTM...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
with open("modelos/scaler_spike.pkl", "wb") as f:
    pickle.dump(scaler, f)

X_lstm = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

print("🔁 Entrenando modelo LSTM...")
model_lstm = Sequential()
model_lstm.add(LSTM(64, input_shape=(X_lstm.shape[1], 1), activation='tanh'))
model_lstm.add(Dense(1, activation='sigmoid'))
model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_lstm.fit(X_lstm, y, epochs=10, batch_size=32, verbose=0)
model_lstm.save("modelos/model_lstm_spike.h5")
print("✅ Modelo LSTM guardado como model_lstm_spike.h5")

# -------------------- ENTRENAR MODELO CNN VISUAL --------------------
print("🔁 Entrenando modelo CNN Visual...")
X_cnn = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1, 1))

model_cnn = Sequential([
    InputLayer(input_shape=(9, 1, 1)),
    Conv2D(filters=32, kernel_size=(3, 1), activation='relu'),
    MaxPooling2D(pool_size=(2, 1)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_cnn.fit(X_cnn, y, epochs=10, batch_size=32, verbose=0)
model_cnn.save("modelos/model_scs_vision_x.keras")
print("✅ Modelo CNN guardado como model_scs_vision_x.keras")
