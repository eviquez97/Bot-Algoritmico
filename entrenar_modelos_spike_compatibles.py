# entrenar_modelos_spike_compatibles.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, Input
from tensorflow.keras.callbacks import EarlyStopping

# 📌 Cargar y preparar el dataset
df = pd.read_csv("data/dataset_spike_monstruo_limpio.csv")
df = df.dropna()

# Solo las columnas válidas actuales
columnas = [
    "fuerza_cuerpo", "score", "rsi", 
    "momentum", "spread", "fuerza_mecha", "bajistas"
]

X = df[columnas]
y = df["spike"]

# Mezclar antes de entrenar
X, y = shuffle(X, y, random_state=42)

# 🔍 División entrenamiento/prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ⚙️ Escalado
scaler_rf = StandardScaler()
X_train_scaled = scaler_rf.fit_transform(X_train)
X_test_scaled = scaler_rf.transform(X_test)

# ===========================
# ✅ Random Forest
# ===========================
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

print("\n[📊 RF] Evaluación:")
print(classification_report(y_test, rf.predict(X_test_scaled)))

joblib.dump(rf, "modelos/model_spike.pkl")
joblib.dump(scaler_rf, "modelos/scaler_rf_2.pkl")
print("[✅ RF GUARDADO] modelos/model_spike.pkl y scaler_rf_2.pkl")

# ===========================
# ✅ LSTM
# ===========================
X_train_lstm = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

model_lstm = Sequential([
    Input(shape=(1, len(columnas))),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_lstm.fit(
    X_train_lstm, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test_lstm, y_test),
    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
    verbose=1
)

model_lstm.save("modelos/model_lstm_spike.keras")
print("[✅ LSTM GUARDADO] modelos/model_lstm_spike.keras")

# ===========================
# ✅ VISUAL (CNN)
# ===========================
X_train_cnn = X_train_lstm  # mismos datos
X_test_cnn = X_test_lstm

model_cnn = Sequential([
    Input(shape=(1, len(columnas))),
    Conv1D(64, kernel_size=1, activation='relu'),
    MaxPooling1D(pool_size=1),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_cnn.fit(
    X_train_cnn, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test_cnn, y_test),
    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
    verbose=1
)

model_cnn.save("modelos/scs_vision_x_model.keras")
print("[✅ VISUAL GUARDADO] modelos/scs_vision_x_model.keras")
