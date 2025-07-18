# 🔍 modelos/verificar_dataset_drl.py

import pandas as pd

df = pd.read_csv("data/dataset_drl.csv")
print(f"Columnas encontradas: {df.columns.tolist()}")
print(f"Filas disponibles: {len(df)}")
