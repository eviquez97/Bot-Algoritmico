import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input

# === CARGA DE DATOS ===
df = pd.read_csv("data/market_data.csv")  # Asegúrate de mover o renombrar tu archivo original
df["spread"] = df["high"] - df["low"]
df["ema"] = df["close"].ewm(span=5, adjust=False).mean()
df["rsi"] = 100 - (100 / (1 + df["close"].pct_change().rolling(3).mean()))
df["momentum"] = df["close"] - df["close"].shift(4)
df = df.dropna()

# === FEATURES Y TARGET ===
features = ["open", "high", "low", "close", "spread", "ema", "rsi", "momentum"]
X = df[features].values
y = (df["close"].shift(-1) < df["close"] - 3).astype(int)[:-1]
X = X[:-1]

# === ESCALADO ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === SECUENCIAS PARA LSTM ===
X_seq, y_seq = [], []
for i in range(60, len(X_scaled)):
    X_seq.append(X_scaled[i - 60:i])
    y_seq.append(y[i])
X_seq, y_seq = np.array(X_seq), np.array(y_seq)

# === MODELO LSTM ===
model = Sequential()
model.add(Input(shape=(60, 8)))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_seq, y_seq, epochs=15, batch_size=32, verbose=1)

# === GUARDAR MODELO Y SCALER ===
model.save("modelos/model_lstm_v2.keras")
joblib.dump(scaler, "modelos/scaler_cierre_predictivo.pkl")

print("[✅ MODELO ENTRENADO Y GUARDADO] model_lstm_v2.keras")
