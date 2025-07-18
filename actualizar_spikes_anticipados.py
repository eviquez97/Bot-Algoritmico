import pandas as pd

# Cargar el archivo original
archivo_csv = "data/dataset_spike_monstruo_limpio.csv"
df = pd.read_csv(archivo_csv)

# Crear nueva columna que marca los spikes reales detectados por variación
df["spike_real"] = df.apply(lambda row: 1 if row["variacion"] > 0 and row["spike"] == 1 else 0, axis=1)

# Guardar el archivo actualizado
df.to_csv("data/dataset_spike_monstruo_limpio.csv", index=False)
print("✅ Archivo actualizado con columna 'spike_real'")
