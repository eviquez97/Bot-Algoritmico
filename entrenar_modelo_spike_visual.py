import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Conv3D, Flatten, Dense, Input
import joblib

# Cargar dataset limpio
df = pd.read_csv("data/dataset_spike_monstruo_limpio.csv").dropna()

# Columnas visuales modernas
columnas = ['fuerza_cuerpo', 'fuerza_mecha', 'bajistas',
            'rsi', 'momentum', 'spread', 'score', 'ema', 'variacion']
X_raw = df[columnas].values
y = df["spike"].values

# Escalar
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw)

# Crear secuencias tipo imagen
X_seq, y_seq = [], []

for i in range(30, len(X_scaled)):
    chunk = X_scaled[i-30:i]  # (30, 9)
    try:
        x = chunk.reshape(30, 3, 3, 1)  # (30, 3, 3, 1)
        X_seq.append(x)
        y_seq.append(y[i])
    except:
        continue

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

if len(X_seq) == 0:
    raise ValueError("❌ No se generó ninguna secuencia válida para entrenamiento.")

# Separar train/test
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Definir modelo CNN visual (Conv3D)
model = Sequential()
model.add(Input(shape=(30, 3, 3, 1)))
model.add(Conv3D(16, kernel_size=(3, 3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compilar y entrenar
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=1)

# Guardar modelo y scaler
model.save("modelos/scs_vision_x_model.keras")
joblib.dump(scaler, "modelos/scaler_spike_visual.pkl")

print("✅ Modelo visual SPIKE (Conv3D) entrenado y guardado correctamente.")
