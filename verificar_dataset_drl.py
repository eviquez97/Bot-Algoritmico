# verificar_dataset_drl.py

import pandas as pd
import os

ruta = "data/dataset_drl.csv"
if not os.path.exists(ruta):
    print(f"[❌ ERROR] No se encontró {ruta}")
    exit()

df = pd.read_csv(ruta)
total = len(df)
print(f"✅ Dataset cargado: {total} filas")

if "accion" not in df.columns:
    print("[❌ ERROR] No se encuentra la columna 'accion'")
    exit()

conteo = df["accion"].value_counts().sort_index()
print("\n📊 Distribución de acciones:")
for accion, cantidad in conteo.items():
    print(f"  Acción {accion}: {cantidad} filas")

faltantes = [a for a in range(4) if a not in conteo]
if faltantes:
    print(f"\n⚠️ Clases faltantes en el dataset: {faltantes}")
else:
    print("\n✅ El dataset tiene todas las clases (0,1,2,3)")
