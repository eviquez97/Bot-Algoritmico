# core/ia_drl.py

import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import core.estado as estado
from core.smart_compound import obtener_entrada_dinamica
from core.ia_modelos import modelo_drl
from utils.logs import log

estado_drl = {
    "total_operaciones": 0,
    "total_ganadas": 0,
    "total_perdidas": 0,
    "evaluaciones": [],
    "epsilon": 0.5,
}

debug_drl = True
CSV_DATASET_DRL = "data/dataset_drl.csv"
CONTADOR_DRL = 120

try:
    columnas_drl = joblib.load("modelos/columnas_drl.pkl")
    log(f"[📥 COLUMNAS DRL] Columnas cargadas correctamente: {columnas_drl}")
except Exception as e:
    log(f"[❌ ERROR COLUMNAS DRL] No se pudo cargar columnas_drl.pkl: {e}")
    columnas_drl = []

if modelo_drl is not None:
    log("✅ Modelo DRL secuencial cargado desde ia_modelos.")
else:
    log("❌ [DRL] El modelo DRL no se cargó correctamente desde ia_modelos.")

modelo_ganancia_rf = None
scaler_rf = None
contador_predicciones = 0

def cargar_modelo_drl_ganancia():
    global modelo_ganancia_rf, scaler_rf
    try:
        modelo_ganancia_rf = joblib.load("modelos/modelo_ganancia_rf.pkl")
        scaler_rf = joblib.load("modelos/scaler_ganancia_rf.pkl")
        log("✅ Modelo DRL de ganancia cargado correctamente.")
    except Exception as e:
        log(f"[❌ ERROR CARGA GANANCIA] {e}")

def procesar_decision_drl(contexto_60_filas, capital, multiplicadores=[100, 200, 300, 400]):
    global modelo_ganancia_rf, scaler_rf, contador_predicciones

    try:
        if estado.contrato_activo:
            log("[⛔ DRL] Ya hay un contrato activo. Decisión omitida.")
            return _respuesta_vacia()

        df = pd.DataFrame(contexto_60_filas)
        if df.shape[0] < 60:
            log(f"[❌ DRL] Contexto insuficiente. Requiere 60 filas, tiene: {df.shape[0]}")
            return _respuesta_vacia()

        df_validas = df[columnas_drl].dropna()
        if df_validas.shape[0] < 60:
            log("[❌ DRL] No hay suficientes filas válidas (sin NaN) para evaluar decisión.")
            return _respuesta_vacia()

        df_filtrado = df_validas.tail(60)
        X_drl = np.array(df_filtrado[columnas_drl]).reshape(1, 60, len(columnas_drl))
        y_pred = modelo_drl.predict(X_drl, verbose=0)
        score_drl = float(np.max(y_pred[0]))
        accion = int(np.argmax(y_pred[0]))

        fila_final = df_filtrado.iloc[-1]
        columnas_rf = ['score', 'rsi', 'momentum', 'spread']
        X_rf = np.array([fila_final[col] for col in columnas_rf]).reshape(1, -1)

        if modelo_ganancia_rf is None or scaler_rf is None:
            cargar_modelo_drl_ganancia()

        ganancia_esperada = float(modelo_ganancia_rf.predict(scaler_rf.transform(X_rf))[0])
        log(f"[🔬 GANANCIA CRUDA RF] Valor estimado: {ganancia_esperada:.4f}")

        if ganancia_esperada <= 0.1 or np.isnan(ganancia_esperada):
            ganancia_esperada = 2.0
            duracion_estimada = 60
            log("[🛡️ MODO SEGURO] Ganancia inválida, usando valores por defecto.")
        else:
            duracion_estimada = max(int(ganancia_esperada * 2), 10)

        monto, multiplicador = obtener_entrada_dinamica(score_drl, ganancia_esperada)

        if debug_drl:
            log("[📊 DRL DECISIÓN]")
            log(f"  Acción: {accion} | Score: {score_drl:.4f}")
            log(f"  Monto: {monto} | Mult: {multiplicador}")
            log(f"  Ganancia: {ganancia_esperada:.2f} | Duración: {duracion_estimada}s")
            log(f"  Predicción: {y_pred[0]}")

        registro = fila_final.to_dict()
        registro.update({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "score": score_drl,
            "ganancia_estimada": ganancia_esperada,
            "accion": accion,
            "monto": monto,
            "multiplicador": multiplicador,
            "exito": 0,
            "Q0": y_pred[0][0],
            "Q1": y_pred[0][1],
            "Q2": y_pred[0][2],
            "Q3": y_pred[0][3]
        })

        df_registro = pd.DataFrame([registro])
        if os.path.exists(CSV_DATASET_DRL):
            columnas_existentes = pd.read_csv(CSV_DATASET_DRL, nrows=1).columns.tolist()
            df_registro = df_registro[[col for col in columnas_existentes if col in df_registro.columns]]
        df_registro.to_csv(CSV_DATASET_DRL, mode='a', header=not os.path.exists(CSV_DATASET_DRL), index=False)

        contador_predicciones += 1
        if contador_predicciones >= CONTADOR_DRL:
            log("[♻️ DRL AUTO] Reentrenando modelo DRL tras 120 decisiones...")
            _entrenar_modelo_drl()
            contador_predicciones = 0

        return {
            "permitir_entrada": True,
            "monto": monto,
            "multiplicador": multiplicador,
            "score": score_drl,
            "ganancia_esperada": ganancia_esperada,
            "duracion_estimada": duracion_estimada,
            "accion": accion,
        }

    except Exception as e:
        log(f"[❌ ERROR GENERAL DRL] {e}")
        return _respuesta_vacia()

def _respuesta_vacia(score_drl=0.0, accion=0):
    return {
        "permitir_entrada": False,
        "monto": 0.0,
        "multiplicador": 0.0,
        "score": score_drl,
        "ganancia_esperada": 0.0,
        "duracion_estimada": 0,
        "accion": accion,
    }

def _entrenar_modelo_drl():
    try:
        df = pd.read_csv(CSV_DATASET_DRL)
        columnas_requeridas = columnas_drl + ["accion"]
        if not all(col in df.columns for col in columnas_requeridas):
            log(f"[❌ DRL ENTRENAMIENTO] Faltan columnas: {columnas_requeridas}")
            return
        df = df.dropna().tail(200)
        X = df[columnas_drl]
        y = df["accion"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense

        X_train_seq = np.array([X_train.iloc[i:i+60].values for i in range(len(X_train)-60)])
        y_train_seq = y_train.iloc[60:].values
        X_test_seq = np.array([X_test.iloc[i:i+60].values for i in range(len(X_test)-60)])
        y_test_seq = y_test.iloc[60:].values

        model = Sequential([
            LSTM(64, input_shape=(60, len(columnas_drl))),
            Dense(32, activation='relu'),
            Dense(len(set(y)), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train_seq, y_train_seq, epochs=30, validation_data=(X_test_seq, y_test_seq), verbose=0,
                  callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

        model.save("modelos/modelo_drl.keras")
        joblib.dump(columnas_drl, "modelos/columnas_drl.pkl")
        log("[✅ DRL ENTRENADO] Modelo guardado.")
    except Exception as e:
        log(f"[❌ DRL ENTRENAMIENTO] Error: {e}")

