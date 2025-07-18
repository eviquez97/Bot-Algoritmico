# entrenar_modelo_drl.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# Ruta del dataset
CSV_DATASET_DRL = "data/dataset_drl.csv"
MODELO_PATH = "modelos/modelo_drl.keras"
COLUMNAS_PATH = "modelos/columnas_drl.pkl"

# Cargar dataset
print("📥 Cargando dataset DRL...")
df = pd.read_csv(CSV_DATASET_DRL).dropna()
columnas_drl = [col for col in df.columns if col not in ["accion", "timestamp", "ganancia_estimada", "monto", "multiplicador"]]
print(f"✅ Columnas para entrenamiento: {len(columnas_drl)} columnas")

# Separar variables y etiquetas
X = df[columnas_drl]
y = df["accion"]

# División entrenamiento/validación
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear secuencias para LSTM
def crear_secuencias(X_data, y_data, secuencia=60):
    X_seq, y_seq = [], []
    for i in range(len(X_data) - secuencia):
        X_seq.append(X_data.iloc[i:i+secuencia].values)
        y_seq.append(y_data.iloc[i+secuencia])
    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train_seq = crear_secuencias(X_train, y_train)
X_test_seq, y_test_seq = crear_secuencias(X_test, y_test)

print(f"✅ Secuencias creadas: {X_train_seq.shape}")

# Construcción del modelo
print("🧠 Entrenando modelo secuencial DRL...")
model = Sequential([
    Input(shape=(60, len(columnas_drl))),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(len(y.unique()), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento validado
if X_test_seq.shape[0] == 0:
    print("[⚠️ AVISO] No hay suficientes secuencias para validación. Entrenando sin conjunto de validación...")
    model.fit(X_train_seq, y_train_seq, epochs=30, verbose=1,
              callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
else:
    model.fit(X_train_seq, y_train_seq, epochs=30, validation_data=(X_test_seq, y_test_seq), verbose=1,
              callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

# Guardado del modelo y columnas
model.save(MODELO_PATH)
joblib.dump(columnas_drl, COLUMNAS_PATH)
print("✅ Modelo DRL entrenado y guardado con éxito.")

