import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

# Rutas
CSV = "data/dataset_drl.csv"
MODELO_PATH = "modelos/modelo_drl.keras"
COLUMNAS_PATH = "modelos/columnas_drl.pkl"

# Silenciar TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Cargar columnas requeridas
try:
    columnas_drl = joblib.load(COLUMNAS_PATH)
except Exception as e:
    print(f"[❌ ERROR] No se pudo cargar columnas_drl.pkl: {e}")
    exit()

# Cargar modelo
try:
    modelo = tf.keras.models.load_model(MODELO_PATH)
    print("✅ Modelo DRL cargado.")
except Exception as e:
    print(f"[❌ ERROR] No se pudo cargar modelo_drl.keras: {e}")
    exit()

# Cargar CSV
try:
    df = pd.read_csv(CSV)
    print(f"📄 CSV cargado: {len(df)} filas.")
except Exception as e:
    print(f"[❌ ERROR] No se pudo cargar el CSV: {e}")
    exit()

# Verificar columnas
faltantes = [col for col in columnas_drl if col not in df.columns]
if faltantes:
    print(f"[❌ ERROR] Faltan columnas requeridas en el CSV: {faltantes}")
    exit()

# Selección y reshape
try:
    df = df[columnas_drl].dropna()
    if len(df) < 60:
        print(f"[❌ ERROR] Solo hay {len(df)} filas válidas. Se requieren mínimo 60.")
        exit()

    df_input = df.tail(60).values.reshape(1, 60, len(columnas_drl))
except Exception as e:
    print(f"[❌ ERROR] Fallo al preparar datos: {e}")
    exit()

# Predicción
try:
    pred = modelo.predict(df_input, verbose=0)[0]
    accion = int(np.argmax(pred))
    print(f"\n🧠 PREDICCIÓN MODELO DRL")
    print(f"Acción sugerida: {accion} (0=Esperar, 1=PUT)")
    print(f"Distribución de salida: {pred}")
except Exception as e:
    print(f"[❌ ERROR] Fallo en la predicción: {e}")
