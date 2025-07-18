# marcar_spike_anticipado.py

import pandas as pd

RUTA_CSV = "data/dataset_spike_monstruo_limpio.csv"

print("🔍 Cargando dataset...")
df = pd.read_csv(RUTA_CSV)

# Validación mínima
if 'spike_real' not in df.columns:
    print("[❌ ERROR] Falta columna 'spike_real'.")
    exit()

# Asegurar columna 'spike_anticipado'
df['spike_anticipado'] = 0

spikes_marcados = 0
for i in range(4, len(df)):
    if df.loc[i, 'spike_real'] == 1:
        for j in range(1, 5):
            idx = i - j
            if idx < 0:
                continue

            # Criterios mínimos adaptados sin columnas crudas
            vela = df.loc[idx]
            condiciones = [
                vela.get("fuerza_cuerpo", 0) > 0.3,
                vela.get("fuerza_mecha", 0) < 0.4,
                vela.get("momentum", 0) < 0,
                vela.get("spread", 0) < 0.25,
                vela.get("score", 0) < 0.5,
                vela.get("variacion", 0) < 0.015
            ]

            if sum(condiciones) >= 4:
                df.at[idx, 'spike_anticipado'] = 1
                spikes_marcados += 1

print(f"✅ Spike anticipado marcado inteligentemente en {spikes_marcados} velas.")
df.to_csv(RUTA_CSV, index=False)
print("✅ Dataset actualizado correctamente.")
