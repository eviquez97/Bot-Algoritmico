import pandas as pd
import os

DATASET_CIERRE = "data/dataset_cierre_monstruo.csv"
COLUMNAS = [
    "profit", "duracion", "score", "futuro", "porcentaje_bajistas",
    "momentum", "rsi", "ema", "spread", "cerrar"
]

def registrar_dato_cierre(dato: dict):
    """
    Guarda una fila de entrenamiento para el modelo de cierre.
    El valor 'cerrar' debe ser 1 si la operación debía cerrarse, 0 si debía continuar.
    """
    try:
        fila = {col: dato.get(col, 0) for col in COLUMNAS}

        if not os.path.exists(DATASET_CIERRE):
            df = pd.DataFrame([fila])
            df.to_csv(DATASET_CIERRE, index=False)
            print("📦 [DATASET CIERRE] Creado con primera fila.")
        else:
            df = pd.DataFrame([fila])
            df.to_csv(DATASET_CIERRE, mode="a", index=False, header=False)
            print("📦 [DATASET CIERRE] Nueva fila añadida.")

    except Exception as e:
        print(f"[❌ ERROR REGISTRO CIERRE] {e}")
