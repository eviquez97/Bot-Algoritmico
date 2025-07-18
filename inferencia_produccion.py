import pandas as pd
import joblib

# === RUTAS ===
CSV_PATH = "data/dataset_operativo.csv"
MODELO_SPIKE_PATH = "modelos/model_spike.pkl"
SCALER_SPIKE_PATH = "modelos/scaler_spike.pkl"
MODELO_DRL_PATH = "modelos/modelo_drl.pkl"
SCALER_DRL_PATH = "modelos/scaler_drl.pkl"
COLUMNAS_DRL_PATH = "modelos/columnas_drl.pkl"

# === CARGA DE MODELOS Y SCALERS ===
modelo_spike = joblib.load(MODELO_SPIKE_PATH)
scaler_spike = joblib.load(SCALER_SPIKE_PATH)
modelo_drl = joblib.load(MODELO_DRL_PATH)
scaler_drl = joblib.load(SCALER_DRL_PATH)
COLUMNAS_DRL = joblib.load(COLUMNAS_DRL_PATH)

print("[✅ MODELOS CARGADOS]")

# === CARGA DEL CSV ===
df = pd.read_csv(CSV_PATH)
print(f"[📊 CSV CARGADO] Total de filas: {len(df)} | Usando últimas 60")

ultimas = df.tail(60).copy()

# === INFERENCIA SPIKE IA ===
try:
    # Solo usar columnas vistas en entrenamiento
    columnas_spike = ["open", "high", "low", "close", "rsi", "ema", "momentum", "score", "spread"]
    X_spike = ultimas[columnas_spike]
    X_spike = scaler_spike.transform(X_spike)
    pred_spike = modelo_spike.predict(X_spike)
    print(f"[🔮 SPIKE IA] Última predicción: {pred_spike[-1]}")
except Exception as e:
    print(f"[❌ ERROR SPIKE IA] {e}")

# === INFERENCIA DRL ===
try:
    X_drl = ultimas[COLUMNAS_DRL]
    X_drl = scaler_drl.transform(X_drl)
    pred_drl = modelo_drl.predict(X_drl)
    print(f"[🤖 DRL] Última decisión: {pred_drl[-1]}")
except Exception as e:
    print(f"[❌ ERROR DRL] {e}")


