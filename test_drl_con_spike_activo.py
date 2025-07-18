# test_drl_con_spike_activo.py

import pandas as pd
from core.ia_drl import procesar_decision_drl
from utils.logs import log

# Cargar el dataset actualizado
df = pd.read_csv("data/dataset_drl.csv")

# Seleccionar las filas inyectadas con spikes
filas_candidatas = df[(df["rf_spike"] > 0.35) & (df["lstm_spike"] > 0.25) & (df["visual_spike"] > 0.3)]
filas_candidatas = filas_candidatas.sort_values(by="ganancia_estimada", ascending=False)

if filas_candidatas.empty:
    print("❌ No se encontraron filas con spikes leves para test.")
    exit()

# Simular contexto de 60 filas para el DRL, usando justo antes de la candidata + la candidata
index_target = filas_candidatas.index[0]
if index_target < 60:
    print("❌ No hay suficientes filas previas para construir contexto.")
    exit()

contexto_simulado = df.iloc[index_target - 59 : index_target + 1]

print(f"🧪 Ejecutando DRL sobre fila {index_target} con spikes leves...")
print(f"📈 rf_spike: {contexto_simulado.iloc[-1]['rf_spike']}, lstm_spike: {contexto_simulado.iloc[-1]['lstm_spike']}, visual_spike: {contexto_simulado.iloc[-1]['visual_spike']}")
print(f"💰 Ganancia esperada: {contexto_simulado.iloc[-1]['ganancia_estimada']:.2f}")

# Simular capital disponible para test
capital = 500.0
multiplicadores = [100, 200, 300, 400]

# Ejecutar decisión
resultado = procesar_decision_drl(contexto_simulado, capital, multiplicadores)

print("\n📊 RESULTADO DECISIÓN DRL CON SPIKE ACTIVO:")
for clave, valor in resultado.items():
    print(f"🔸 {clave}: {valor}")
