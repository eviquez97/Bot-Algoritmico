# test_spike_manual.py

import pandas as pd
from core.ia_spike import evaluar_spike_ia

CSV = "data/contexto_historico.csv"

print("🔍 Cargando CSV...")
try:
    df = pd.read_csv(CSV)
    print(f"✅ CSV cargado correctamente con {len(df)} filas.")
except Exception as e:
    print(f"❌ Error al cargar el CSV: {e}")
    exit()

# Solo las últimas 60 velas
df_ultimas = df.tail(60).copy()

print("🧠 Ejecutando evaluación Spike IA...")
resultado = evaluar_spike_ia(df_ultimas)

print("\n📊 Resultado de Spike IA:")
for clave, valor in resultado.items():
    print(f"   {clave}: {valor}")
