# entrenar_modelos_spike_anticipado.py

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Cargar dataset
df = pd.read_csv("data/dataset_spike_monstruo_limpio.csv")

# Asegurarse de no tener NaNs
df = df.dropna()

# Definir columnas para cada modelo
columnas_rf_lstm = ['fuerza_cuerpo', 'fuerza_mecha', 'bajistas', 'rsi', 'momentum', 'spread', 'score', 'variacion']
columnas_visual = columnas_rf_lstm + ['ema']

# ============================
# 🎯 MODELO 1: Random Forest
# ============================

X_rf = df[columnas_rf_lstm]
y_rf = df["spike_anticipado"]

scaler_rf = StandardScaler()
X_rf_scaled = scaler_rf.fit_transform(X_rf)

modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_rf_scaled, y_rf)

joblib.dump(modelo_rf, "modelos/model_spike.pkl")
joblib.dump(scaler_rf, "modelos/scaler_rf_2.pkl")

print("✅ Modelo Random Forest entrenado y guardado.")

# ============================
# 🎯 MODELO 2: LSTM
# ============================

# Preparar datos para LSTM: secuencia de 30 velas
sequence_length = 30
X_seq = []
y_seq = []

for i in range(sequence_length, len(X_rf_scaled)):
    X_seq.append(X_rf_scaled[i-sequence_length:i])
    y_seq.append(y_rf.values[i])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

modelo_lstm = Sequential()
modelo_lstm.add(LSTM(64, input_shape=(X_seq.shape[1], X_seq.shape[2])))
modelo_lstm.add(Dense(1, activation='sigmoid'))
modelo_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

modelo_lstm.fit(X_train, y_train, epochs=10, batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])

modelo_lstm.save("modelos/model_lstm_spike.keras")
print("✅ Modelo LSTM entrenado y guardado.")

# ============================
# 🎯 MODELO 3: Visual / CNN o secuencial básico
# ============================

X_vis = df[columnas_visual].values
y_vis = df["spike_anticipado"].values

# Escalar
scaler_vis = StandardScaler()
X_vis_scaled = scaler_vis.fit_transform(X_vis)

# Secuencias para visual (30 velas)
X_vis_seq = []
y_vis_seq = []

for i in range(sequence_length, len(X_vis_scaled)):
    X_vis_seq.append(X_vis_scaled[i-sequence_length:i])
    y_vis_seq.append(y_vis[i])

X_vis_seq = np.array(X_vis_seq)
y_vis_seq = np.array(y_vis_seq)

X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(X_vis_seq, y_vis_seq, test_size=0.2, random_state=42)

modelo_vis = Sequential()
modelo_vis.add(LSTM(64, input_shape=(X_vis_seq.shape[1], X_vis_seq.shape[2])))
modelo_vis.add(Dense(1, activation='sigmoid'))
modelo_vis.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

modelo_vis.fit(X_train_v, y_train_v, epochs=10, batch_size=32,
               validation_data=(X_test_v, y_test_v),
               callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])

modelo_vis.save("modelos/scs_vision_x_model.keras")
print("✅ Modelo visual entrenado y guardado.")
