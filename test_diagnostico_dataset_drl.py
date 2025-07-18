# test_diagnostico_dataset_drl.py

import pandas as pd
import numpy as np
from collections import Counter

CSV_PATH = "data/dataset_drl.csv"

try:
    df = pd.read_csv(CSV_PATH)
    print(f"✅ Dataset cargado: {len(df)} filas\n")
except Exception as e:
    print(f"❌ Error al cargar el dataset: {e}")
    exit()

# -------- Diagnóstico General --------
print("📊 Diagnóstico general del dataset:")

# 1. NaNs
total_nans = df.isna().sum().sum()
filas_con_nans = df.isna().any(axis=1).sum()
print(f"🔍 Filas con NaN: {filas_con_nans} / {len(df)}")
print(f"🔍 Total de valores NaN: {total_nans}")

# 2. Estadísticas clave
campos_clave = ["score", "rsi", "momentum", "spread", "ganancia_estimada"]
for campo in campos_clave:
    if campo in df.columns:
        print(f"📈 {campo}: min={df[campo].min()}, max={df[campo].max()}, mean={df[campo].mean():.2f}")
    else:
        print(f"⚠️ Campo faltante: {campo}")

# 3. Clases disponibles en "accion"
if "accion" in df.columns:
    distribucion = dict(Counter(df["accion"]))
    print(f"\n🔢 Distribución de clases en 'accion': {distribucion}")
    if len(distribucion) < 2:
        print("⚠️ Solo hay una clase presente. El modelo no podrá aprender a diferenciar acciones.")
else:
    print("❌ No existe la columna 'accion' en el dataset.")

# 4. Recomendación
if filas_con_nans > 0:
    print("\n⚠️ Hay filas incompletas. Se recomienda limpiar el dataset antes de reentrenar.")
else:
    print("\n✅ Dataset limpio. Apto para reentrenamiento si hay variedad de acciones.")

