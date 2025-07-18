import os
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from utils.logs import log
from reentrenamiento.entrenador_spike import reentrenar_modelo_spike
from reentrenamiento.entrenador_cierre import entrenar_modelo_cierre
from core.ia_drl_entrenamiento import reentrenar_modelo_drl

DATASET_PATH = "data/contexto_historico.csv"
MODELO_RF_PATH = "modelos/model_spike.pkl"
SCALER_RF_PATH = "modelos/scaler_rf_2.pkl"
MODELO_LSTM_PATH = "modelos/model_lstm_futuro.keras"
SCALER_LSTM_PATH = "modelos/scaler_futuro.pkl"

def entrenar_modelos():
    if not os.path.exists(DATASET_PATH):
        log("❌ No existe el dataset para reentrenamiento.")
        return

    df = pd.read_csv(DATASET_PATH).dropna()

    if len(df) < 100:
        log("⚠️ Dataset muy pequeño para reentrenar modelos principales.")
        return

    columnas = ['open', 'high', 'low', 'close', 'spread', 'ema', 'rsi', 'momentum']
    if not all(col in df.columns for col in columnas):
        log(f"❌ Faltan columnas necesarias para entrenamiento: {columnas}")
        return

    # Reentrenamiento RandomForest
    X_rf = df[columnas]
    y_rf = df["close"].shift(-1) < df["close"]
    X_rf = X_rf[:-1]
    y_rf = y_rf[:-1]

    scaler_rf = StandardScaler()
    X_rf_scaled = scaler_rf.fit_transform(X_rf)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_rf_scaled, y_rf)

    joblib.dump(rf_model, MODELO_RF_PATH)
    joblib.dump(scaler_rf, SCALER_RF_PATH)
    log("✅ [RF] Modelo RandomForest futuro reentrenado correctamente.")

    # Reentrenamiento LSTM
    secuencias = []
    for i in range(30, len(df)):
        ventana = df[columnas].iloc[i-30:i].values
        secuencias.append(ventana)

    X_lstm = np.array(secuencias)
    y_lstm = df["close"].shift(-1).iloc[30:].values

    scaler_lstm = StandardScaler()
    X_lstm_reshaped = X_lstm.reshape(-1, len(columnas))
    X_scaled = scaler_lstm.fit_transform(X_lstm_reshaped).reshape(X_lstm.shape)

    model = Sequential()
    model.add(LSTM(64, input_shape=(30, len(columnas))))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_scaled, y_lstm, epochs=10, batch_size=32, verbose=0,
              callbacks=[EarlyStopping(monitor='loss', patience=2)])

    model.save(MODELO_LSTM_PATH)
    joblib.dump(scaler_lstm, SCALER_LSTM_PATH)
    log("✅ [LSTM] Modelo de predicción futura reentrenado correctamente.")

    # Reentrenamiento DRL, SPIKE y CIERRE
    reentrenar_modelo_drl()
    reentrenar_modelo_spike()
    entrenar_modelo_cierre()

def ejecutar_si_es_hora():
    ahora = datetime.now()
    if ahora.hour == 23 and ahora.minute == 30:
        log("[⏰ AUTOENTRENAMIENTO DIARIO] Ejecutando entrenamiento programado...")
        entrenar_modelos()


