import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import joblib

# Rutas
RUTA_CSV = "data/dataset_spike_monstruo_limpio.csv"
RUTA_MODELO = "modelos/scs_vision_x_model.keras"
RUTA_SCALER = "modelos/scaler_visual_spike.pkl"

# Columnas de entrada
columnas = ['fuerza_cuerpo', 'fuerza_mecha', 'bajistas', 'rsi', 'momentum', 'spread', 'score', 'ema', 'variacion']

# 1. Cargar dataset
df = pd.read_csv(RUTA_CSV)
df = df.dropna()
df = df[df["spike_anticipado"].isin([0, 1])]

# 2. Escalar y preparar datos
scaler = MinMaxScaler()
X_escalado = scaler.fit_transform(df[columnas])
joblib.dump(scaler, RUTA_SCALER)

# 3. Reconstruir ventanas de 32 velas para entrada CNN
ventana = 32
X_seq = []
y_seq = []

for i in range(len(X_escalado) - ventana):
    X_seq.append(X_escalado[i:i+ventana])
    y_seq.append(df["spike_anticipado"].iloc[i + ventana])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# Balanceo opcional (puede omitir esta parte si ya está balanceado)
# ...

# 4. Separar conjuntos
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq)

# 5. Crear modelo CNN
modelo = Sequential([
    Input(shape=(ventana, len(columnas))),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Conv1D(128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binaria
])

modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. Entrenar
modelo.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
    verbose=1
)

# 7. Guardar modelo
modelo.save(RUTA_MODELO)
print("✅ Modelo Visual CNN entrenado y guardado correctamente.")
