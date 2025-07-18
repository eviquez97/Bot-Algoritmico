# test_verificar_fallo_spike.py

import pandas as pd
import joblib
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

CSV_PATH = "data/dataset_spike_monstruo_limpio.csv"
MODEL_RF = "modelos/model_spike.pkl"
MODEL_LSTM = "modelos/model_lstm_spike.h5"
SCALER_PATH = "modelos/scaler_spike_rf.pkl"

# 1. Cargar dataset
df = pd.read_csv(CSV_PATH)

# 2. Buscar última fila con spike_real = 1
if "spike_real" in df.columns:
    ult_spike = df[df["spike_real"] == 1].tail(1)
    if not ult_spike.empty:
        idx = ult_spike.index[0]
        print(f"\n🔎 Último spike real en fila: {idx}")

        # Mostrar las 2 velas anteriores
        prev_rows = df.iloc[idx-2:idx]
        print("🕒 Velas anteriores (esperadas como anticipadas):")
        print(prev_rows)

        # Preparar datos para modelos
        columnas = ['fuerza_cuerpo', 'fuerza_mecha', 'bajistas', 'rsi', 'momentum', 'spread', 'score', 'ema', 'variacion']
        X_test = df.loc[idx-2:idx-1, columnas]

        # RandomForest
        modelo_rf = joblib.load(MODEL_RF)
        scaler = joblib.load(SCALER_PATH)
        X_scaled = scaler.transform(X_test)
        pred_rf = modelo_rf.predict_proba(X_scaled)[:, 1]
        print(f"\n🎯 RF Predicciones anticipadas: {[round(float(p), 2) for p in pred_rf]}")

        # LSTM
        modelo_lstm = load_model(MODEL_LSTM)
        X_lstm = X_test.values.astype("float32")  # CORREGIDO: No reshape
        pred_lstm = modelo_lstm.predict(X_lstm)
        print(f"🎯 LSTM Predicciones anticipadas: {[round(float(p), 2) for p in pred_lstm]}")
    else:
        print("❌ No se encontraron spikes reales en el CSV.")
else:
    print("❌ El CSV no contiene la columna 'spike_real'.")
