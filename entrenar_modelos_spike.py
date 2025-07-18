# entrenar_modelos_spike.py

import pandas as pd
import numpy as np
import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# ⚙️ Ruta al dataset limpio
DATASET = "data/dataset_spike_monstruo_limpio.csv"

# 📥 Carga del CSV
df = pd.read_csv(DATASET).dropna()
print(f"✅ Dataset cargado: {len(df)} filas")

# 🎯 Features y target
X = df.drop(columns=["spike"])
y = df["spike"]

# 🧪 División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ========== 🎯 1. RANDOM FOREST ==========
scaler_rf = StandardScaler()
X_train_scaled = scaler_rf.fit_transform(X_train)
X_test_scaled = scaler_rf.transform(X_test)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 💾 Guardado
with open("modelos/model_spike.pkl", "wb") as f:
    pickle.dump(rf_model, f)
with open("modelos/scaler_rf_2.pkl", "wb") as f:
    pickle.dump(scaler_rf, f)

print("✅ Modelo RF entrenado y guardado.")

# ========== 🧠 2. LSTM ==========
# Convierte a formato (samples, timesteps, features)
X_lstm = np.array(X).reshape((X.shape[0], 1, X.shape[1]))
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y, test_size=0.2, stratify=y, random_state=42)

lstm_model = Sequential()
lstm_model.add(LSTM(64, input_shape=(1, X.shape[1]), activation="relu"))
lstm_model.add(Dense(1, activation="sigmoid"))

lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

lstm_model.fit(X_train_lstm, y_train_lstm, epochs=30, batch_size=16, validation_split=0.2, callbacks=[early_stop], verbose=0)

lstm_model.save("modelos/model_lstm_spike.keras")
print("✅ Modelo LSTM entrenado y guardado.")

# ========== 🧠 3. VISUAL (CNN 1D SIMPLIFICADA) ==========
# Similar al LSTM, usamos reshape
X_cnn = np.array(X).reshape((X.shape[0], 1, X.shape[1]))
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X_cnn, y, test_size=0.2, stratify=y, random_state=42)

from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten

visual_model = Sequential()
visual_model.add(Conv1D(filters=32, kernel_size=1, activation='relu', input_shape=(1, X.shape[1])))
visual_model.add(MaxPooling1D(pool_size=1))
visual_model.add(Flatten())
visual_model.add(Dense(50, activation='relu'))
visual_model.add(Dense(1, activation='sigmoid'))

visual_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

visual_model.fit(X_train_cnn, y_train_cnn, epochs=30, batch_size=16, validation_split=0.2, callbacks=[early_stop], verbose=0)

visual_model.save("modelos/scs_vision_x_model.keras")
print("✅ Modelo VISUAL CNN entrenado y guardado.")
