# test_modelo_spike_rf.py

import pandas as pd
import numpy as np
from joblib import load
from sklearn.exceptions import NotFittedError

# Rutas
RUTA_CSV = "data/dataset_spike_monstruo_limpio.csv"
RUTA_MODELO = "modelos/model_spike.pkl"
RUTA_SCALER = "modelos/scaler_rf_2.pkl"

# Cargar dataset y modelos
df = pd.read_csv(RUTA_CSV)
modelo = load(RUTA_MODELO)
scaler = load(RUTA_SCALER)

# Columnas requeridas
columnas = list(scaler.feature_names_in_)

# Preparamos último bloque de 30 velas válidas
df = df.dropna()
df = df[columnas].tail(30)

if len(df) < 30:
    print(f"❌ No hay suficientes velas válidas. Solo hay {len(df)}")
else:
    try:
        x = scaler.transform(df)
        print("📐 Shape del input al modelo:", x.shape)

        # Predicción directa
        y_pred = modelo.predict(x)
        print("🔍 Predicciones crudas:", y_pred)

        # Última predicción
        print(f"🧠 Última predicción RF Spike:", round(float(y_pred[-1]), 4))

    except NotFittedError:
        print("❌ El modelo no está entrenado.")
    except Exception as e:
        print(f"❌ Error durante la predicción: {e}")
