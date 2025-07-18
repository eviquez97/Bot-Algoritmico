# entrenar_modelo_drl.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Ruta del CSV exportado
CSV_PATH = "data/dataset_drl.csv"

# Columnas válidas que se utilizarán como entrada
columnas_drl_validas = [
    "score", "rsi", "momentum", "spread",
    "ema", "variacion", "fuerza_cuerpo", "fuerza_mecha",
    "mecha_superior", "mecha_inferior", "bajistas"
]

# Cargar y filtrar dataset
df = pd.read_csv(CSV_PATH)
df = df[columnas_drl_validas + ["accion"]].dropna()

X = df[columnas_drl_validas]
y = df["accion"].astype(int)
num_classes = len(np.unique(y))

# Separar entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Modelo secuencial
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento
model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=30,
    verbose=1,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
)

# Evaluación
loss, acc = model.evaluate(X_test, y_test_cat)
print(f"✅ Modelo entrenado | Precisión: {acc:.4f} | Pérdida: {loss:.4f}")

# Guardar modelo y columnas
model.save("modelos/modelo_drl.keras")
joblib.dump(columnas_drl_validas, "modelos/columnas_drl.pkl")
print("📦 Modelo y columnas guardados en carpeta modelos/")
