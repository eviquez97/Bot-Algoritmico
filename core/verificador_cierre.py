import os
import pandas as pd
import datetime
import joblib
from utils.logs import log

def verificar_integridad_cierre():
    hoy = datetime.date.today()
    errores = []

    # Rutas
    modelo_path = "modelos/model_cierre.pkl"
    scaler_path = "modelos/scaler_cierre.pkl"
    dataset_path = "data/dataset_cierre.csv"

    # Verificación de archivos
    if not os.path.exists(modelo_path):
        errores.append("❌ Falta el modelo model_cierre.pkl")

    if not os.path.exists(scaler_path):
        errores.append("❌ Falta el scaler scaler_cierre.pkl")

    if not os.path.exists(dataset_path):
        errores.append("❌ Falta el dataset dataset_cierre.csv")

    # Fechas de actualización
    if os.path.exists(modelo_path):
        fecha_modelo = datetime.date.fromtimestamp(os.path.getmtime(modelo_path))
        if fecha_modelo != hoy:
            errores.append("⚠️ El modelo de cierre no fue actualizado hoy")

    if os.path.exists(scaler_path):
        fecha_scaler = datetime.date.fromtimestamp(os.path.getmtime(scaler_path))
        if fecha_scaler != hoy:
            errores.append("⚠️ El scaler de cierre no fue actualizado hoy")

    # Validación del dataset
    if os.path.exists(dataset_path):
        try:
            df = pd.read_csv(dataset_path)
            columnas_requeridas = ['open', 'high', 'low', 'close', 'spread', 'ema', 'rsi', 'momentum', 'resultado']

            faltantes = [col for col in columnas_requeridas if col not in df.columns]
            if faltantes:
                errores.append(f"❌ Faltan columnas requeridas: {faltantes}")

            if len(df) < 50:
                errores.append(f"⚠️ Dataset de cierre con pocas filas: {len(df)} (< 50)")

            ultimas_30 = df[columnas_requeridas].dropna().tail(30)
            if len(ultimas_30) < 30:
                errores.append("⚠️ Las últimas 30 filas del dataset de cierre tienen valores nulos o insuficientes")

        except Exception as e:
            errores.append(f"❌ Error al leer o procesar el dataset de cierre: {e}")

    # Log final
    if errores:
        log("[🧪 VERIFICACIÓN CIERRE] Fallos detectados:")
        for error in errores:
            log(error)
    else:
        log("✅ [VERIFICACIÓN CIERRE] Modelo, scaler y dataset de cierre están en orden.")
