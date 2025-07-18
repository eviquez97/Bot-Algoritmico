import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten
import joblib
import os

# --- Configuración
RUTA_CSV = "data/dataset_spike_monstruo_limpio.csv"
COLUMNAS = ['fuerza_cuerpo', 'fuerza_mecha', 'bajistas', 'rsi', 'momentum', 'spread', 'score', 'ema', 'variacion']
ETIQUETA = 'spike_anticipado'

# --- Carga de datos
df = pd.read_csv(RUTA_CSV)
df = df[COLUMNAS + [ETIQUETA]].dropna()
X = df[COLUMNAS]
y = df[ETIQUETA]

# --- Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "modelos/scaler_spike_rf.pkl")  # para RF
np.save("modelos/columnas_spike.npy", np.array(COLUMNAS))

# --- Modelo 1: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)
joblib.dump(rf, "modelos/model_spike.pkl")
print("[✅ RF] Entrenamiento completado")

# --- Preparación para modelos secuenciales
residuo = X_scaled.shape[0] % 30
if residuo != 0:
    X_scaled = X_scaled[:-residuo]
    y = y[:-residuo]

X_seq = np.reshape(X_scaled, (-1, 30, len(COLUMNAS)))
y_seq = y.tail(X_seq.shape[0]).values

X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# --- Modelo 2: LSTM
model_lstm = Sequential([
    LSTM(64, input_shape=(30, len(COLUMNAS))),
    Dense(1, activation='sigmoid')
])
model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_lstm.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
model_lstm.save("modelos/model_lstm_spike.keras")
print("[✅ LSTM] Entrenamiento completado")

# --- Modelo 3: Visual (CNN)
model_cnn = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(30, len(COLUMNAS))),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(1, activation='sigmoid')
])
model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_cnn.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
model_cnn.save("modelos/scs_vision_x_model.keras")
print("[✅ VISUAL] Entrenamiento completado")

# --- Reporte
y_pred = rf.predict(X_scaled)
print("[📊 RF] Reporte de clasificación:")
print(classification_report(y, y_pred))
