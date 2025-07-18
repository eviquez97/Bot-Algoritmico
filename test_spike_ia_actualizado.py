# test_spike_ia_actualizado.py

import pandas as pd
from core.ia_spike import evaluar_spike_ia

# Cargar contexto histórico
df = pd.read_csv("data/contexto_historico.csv")

# Usar últimas 60 velas
df_test = df.tail(60).copy()

# Ejecutar evaluación
print("🔍 Ejecutando evaluación de SPIKE IA...")
resultado = evaluar_spike_ia(df_test)

# Mostrar resultado completo
print("\n📊 Resultado retornado:")
for k, v in resultado.items():
    print(f"{k}: {v}")
