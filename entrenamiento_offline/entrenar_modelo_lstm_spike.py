# entrenar_modelo_lstm_spike.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
import joblib

# ✅ Cargar el dataset limpio
CSV_PATH = "data/dataset_spike_monstruo_limpio.csv"
df = pd.read_csv(CSV_PATH)

# ✅ Columnas modernas utilizadas por el bot
columnas_usadas = ['fuerza_cuerpo', 'fuerza_mecha', 'bajistas', 'rsi', 'momentum',
                   'spread', 'score', 'ema', 'variacion']
target = 'spike'

# ✅ Filtrar columnas y eliminar nulos
df = df[columnas_usadas + [target]].dropna()

# ✅ Normalizar entradas
scaler = MinMaxScaler()
X = scaler.fit_transform(df[columnas_usadas])
y = df[target].astype(int).values

# ✅ Guardar el scaler
joblib.dump(scaler, "modelos/scaler_rf_2.pkl")

# ✅ Preparar datos para LSTM (reshape a 3D)
X_lstm = X.reshape((X.shape[0], 1, X.shape[1]))

# ✅ Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y, test_size=0.2, random_state=42)

# ✅ Construir modelo LSTM
model = Sequential()
model.add(LSTM(64, input_shape=(1, len(columnas_usadas))))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ✅ Entrenar modelo
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, callbacks=[early_stop], verbose=1)

# ✅ Guardar modelo
model.save("modelos/model_lstm_spike.h5")
print("✅ Modelo LSTM SPIKE entrenado y guardado correctamente.")
