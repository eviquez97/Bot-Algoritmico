# limpiar_dataset_drl_sobrescribir.py

import pandas as pd

ARCHIVO = "data/dataset_drl.csv"

try:
    df = pd.read_csv(ARCHIVO)
    original = len(df)
    df = df.dropna()
    limpio = len(df)

    df.to_csv(ARCHIVO, index=False)
    print(f"✅ Dataset actualizado correctamente: {ARCHIVO}")
    print(f"🧮 Filas originales: {original} | Filas limpias: {limpio} | Eliminadas: {original - limpio}")
except Exception as e:
    print(f"❌ Error al limpiar el dataset: {e}")
