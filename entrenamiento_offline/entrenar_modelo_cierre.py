import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# ===============================
# 📥 Cargar y verificar CSV
# ===============================
csv_path = "data/dataset_entrenamiento_cierre.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError("❌ No se encontró el archivo CSV requerido para entrenar.")

df = pd.read_csv(csv_path)

# ===============================
# ✅ Validar columnas necesarias
# ===============================
columnas_requeridas = ["open", "high", "low", "close", "target"]
faltantes = [col for col in columnas_requeridas if col not in df.columns]
if faltantes:
    raise ValueError(f"❌ Faltan columnas obligatorias en el CSV: {faltantes}")

# ===============================
# 🧪 Ingeniería de características
# ===============================
df["spread"] = df["high"] - df["low"]
df["momentum"] = df["close"].diff()
df["ema"] = df["close"].ewm(span=10, adjust=False).mean()
df["rsi"] = 100 - (100 / (1 + (df["close"].diff().clip(lower=0).rolling(14).mean() /
                              -df["close"].diff().clip(upper=0).rolling(14).mean())))
df["variacion"] = (df["close"] - df["open"]) / df["open"]
df["score"] = df["variacion"].rolling(5).mean()
df["target"] = df["target"].fillna(0)

# ===============================
# 🔍 Validar NaN y preparar datos
# ===============================
columnas_entrada = ["open", "high", "low", "close", "spread", "momentum", "ema", "rsi", "variacion", "score"]
df = df.dropna(subset=columnas_entrada + ["target"])

if len(df) < 100:
    raise ValueError("❌ No hay suficientes datos válidos para entrenar el modelo de cierre.")

X_df = df[columnas_entrada].copy()
y = df["target"]

scaler = MinMaxScaler()
scaler.fit(X_df)
X = scaler.transform(X_df)

# ===============================
# ⏱️ Secuencias LSTM (30 pasos)
# ===============================
def crear_secuencias(X, y, pasos=30):
    Xs, ys = [], []
    for i in range(len(X) - pasos):
        Xs.append(X[i:i+pasos])
        ys.append(y.iloc[i+pasos])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = crear_secuencias(X, y)

# ===============================
# 🧠 Entrenamiento del modelo
# ===============================
X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32)

# ===============================
# 💾 Guardar modelo y scaler
# ===============================
os.makedirs("modelos", exist_ok=True)
model.save("modelos/model_cierre.keras")
joblib.dump(scaler, "modelos/scaler_cierre.pkl")

print("✅ Modelo LSTM de cierre entrenado y guardado correctamente.")

