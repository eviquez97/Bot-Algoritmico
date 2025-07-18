# verificador_spike.py
import os
import pandas as pd
import datetime
import joblib
from keras.models import load_model
from utils.logs import log

def verificar_integridad_spike():
    hoy = datetime.date.today()
    errores = []

    # 📁 Paths
    modelo_path = "modelos/model_lstm_spike.h5"
    scaler_path = "modelos/scaler_lstm_spike.pkl"
    dataset_path = "data/dataset_operativo.csv"  # ✅ Usamos el dataset correcto

    # 📌 Verificación de existencia de archivos
    if not os.path.exists(modelo_path):
        errores.append("❌ Falta el modelo model_lstm_spike.h5")

    if not os.path.exists(scaler_path):
        errores.append("❌ Falta el scaler scaler_lstm_spike.pkl")

    if not os.path.exists(dataset_path):
        errores.append("❌ Falta el dataset dataset_operativo.csv")

    # 📅 Validación de fecha de actualización
    if os.path.exists(modelo_path):
        fecha_modelo = datetime.date.fromtimestamp(os.path.getmtime(modelo_path))
        if fecha_modelo != hoy:
            errores.append("⚠️ El modelo no fue actualizado hoy")

    if os.path.exists(scaler_path):
        fecha_scaler = datetime.date.fromtimestamp(os.path.getmtime(scaler_path))
        if fecha_scaler != hoy:
            errores.append("⚠️ El scaler no fue actualizado hoy")

    # 🧪 Validación interna del dataset
    if os.path.exists(dataset_path):
        try:
            df = pd.read_csv(dataset_path)
            columnas_requeridas = ['open', 'high', 'low', 'close', 'spread', 'ema', 'rsi', 'momentum', 'target']
            columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]

            if columnas_faltantes:
                errores.append(f"❌ Faltan columnas requeridas: {columnas_faltantes}")

            if len(df) < 50:
                errores.append(f"⚠️ Muy pocos datos en el dataset: {len(df)} filas (< 50)")

            ultimas_30 = df[columnas_requeridas].dropna().tail(30)
            if len(ultimas_30) < 30:
                errores.append("⚠️ Hay valores nulos o insuficientes en las últimas 30 filas del dataset")

        except Exception as e:
            errores.append(f"❌ Error al leer o procesar el dataset: {e}")

    # ✅ Log final
    if errores:
        log("[🧪 VERIFICACIÓN SPIKE] Fallos detectados:")
        for error in errores:
            log(error)
    else:
        log("✅ [VERIFICACIÓN SPIKE] Todo correcto: modelo, scaler y dataset actualizados y válidos.")


