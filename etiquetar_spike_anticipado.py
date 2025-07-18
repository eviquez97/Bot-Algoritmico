import pandas as pd

RUTA_CSV = "data/dataset_spike_monstruo_limpio.csv"

# Cargar el dataset
df = pd.read_csv(RUTA_CSV)

# Asegurarse de que la columna objetivo existe
if "spike_real" not in df.columns:
    raise Exception("❌ La columna 'spike_real' no existe en el CSV.")

# Crear la columna si no existe
if "spike_anticipado" not in df.columns:
    df["spike_anticipado"] = 0

# Resetear la columna por si ya estaba parcialmente marcada
df["spike_anticipado"] = 0

# Recorremos cada fila que tenga spike_real = 1
for idx in df.index[df["spike_real"] == 1]:
    # Etiquetar las 2 velas anteriores como anticipadas
    for anticipada in [idx - 1, idx - 2]:
        if anticipada >= 0:
            df.at[anticipada, "spike_anticipado"] = 1

# Guardar el archivo con las etiquetas aplicadas
df.to_csv(RUTA_CSV, index=False)
print("✅ Etiquetado de spike_anticipado completado y guardado correctamente.")
