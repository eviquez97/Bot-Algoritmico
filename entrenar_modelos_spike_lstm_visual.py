import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# === CONFIG ===
CSV_PATH = "data/dataset_spike_monstruo_limpio.csv"
SCALER_PATH = "modelos/scaler_lstm_spike.pkl"
LSTM_MODEL_PATH = "modelos/model_lstm_spike.keras"
CNN_MODEL_PATH = "modelos/scs_vision_x_model.keras"

# === CARGA DATASET ===
df = pd.read_csv(CSV_PATH).dropna()
features = ['fuerza_cuerpo', 'fuerza_mecha', 'bajistas', 'rsi', 'momentum', 'spread', 'score', 'ema', 'variacion']
X = df[features].values
y = df["spike"].values

# === ESCALAR ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, SCALER_PATH)

# === CREAR SECUENCIAS PARA LSTM ===
def crear_secuencias(X, y, secuencia=30):
    Xs, ys = [], []
    for i in range(len(X) - secuencia):
        Xs.append(X[i:i+secuencia])
        ys.append(y[i+secuencia])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = crear_secuencias(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq)

# === ENTRENAR MODELO LSTM ===
model_lstm = Sequential()
model_lstm.add(LSTM(64, input_shape=(X_seq.shape[1], X_seq.shape[2]), return_sequences=False))
model_lstm.add(Dropout(0.3))
model_lstm.add(Dense(1, activation='sigmoid'))
model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_lstm.fit(X_train, y_train, epochs=20, batch_size=16,
               validation_data=(X_test, y_test),
               callbacks=[EarlyStopping(patience=3)], verbose=1)
model_lstm.save(LSTM_MODEL_PATH)
print(f"[✅ MODELO LSTM GUARDADO] {LSTM_MODEL_PATH}")

# === ENTRENAR MODELO VISUAL CNN ===
X_img = X_scaled.reshape(-1, 3, 3, 1)
y_img = y[:len(X_img)]

X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(X_img, y_img, test_size=0.2, random_state=42, stratify=y_img)

model_cnn = Sequential()
model_cnn.add(Conv2D(32, (2, 2), activation='relu', input_shape=(3, 3, 1)))
model_cnn.add(MaxPooling2D((1, 1)))
model_cnn.add(Flatten())
model_cnn.add(Dense(32, activation='relu'))
model_cnn.add(Dropout(0.3))
model_cnn.add(Dense(1, activation='sigmoid'))
model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_cnn.fit(X_train_img, y_train_img, epochs=20, batch_size=16,
              validation_data=(X_test_img, y_test_img),
              callbacks=[EarlyStopping(patience=3)], verbose=1)
model_cnn.save(CNN_MODEL_PATH)
print(f"[✅ MODELO VISUAL GUARDADO] {CNN_MODEL_PATH}")
