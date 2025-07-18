import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# ====== Paso 1: Generar datos sintéticos de velas ======
np.random.seed(42)
n = 500
df = pd.DataFrame({
    'open': np.random.uniform(10000, 20000, n),
    'high': np.random.uniform(10000, 20000, n),
    'low': np.random.uniform(10000, 20000, n),
    'close': np.random.uniform(10000, 20000, n),
    'spread': np.random.uniform(0.1, 2.0, n),
    'momentum': np.random.uniform(-1.0, 1.0, n),
    'ema': np.random.uniform(10000, 20000, n),
    'rsi': np.random.uniform(0, 100, n),
    'target': np.random.randint(0, 2, n)
})

# ====== Paso 2: Escalar ======
X = df.drop(columns=['target'])
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ====== Paso 3: Formatear para LSTM ======
ventana = 10
X_lstm = []
y_lstm = []

for i in range(ventana, len(X_scaled)):
    X_lstm.append(X_scaled[i-ventana:i])
    y_lstm.append(y.iloc[i])

X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

# ====== Paso 4: Definir y entrenar modelo LSTM ======
model = Sequential()
model.add(LSTM(64, input_shape=(X_lstm.shape[1], X_lstm.shape[2]), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_lstm, y_lstm, epochs=10, batch_size=32, verbose=1,
          callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])

# ====== Paso 5: Guardar modelo y scaler ======
model.save('modelos/model_lstm_spike.h5')
joblib.dump(scaler, 'modelos/scaler_lstm_spike.pkl')

print("✅ Modelo guardado como 'model_lstm_spike.h5' y scaler como 'scaler_lstm_spike.pkl'")
