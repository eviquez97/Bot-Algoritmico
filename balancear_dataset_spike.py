import pandas as pd
from sklearn.utils import shuffle

RUTA_CSV = "data/dataset_spike_monstruo_limpio.csv"

print("🔍 Cargando dataset original...")
df = pd.read_csv(RUTA_CSV)

# Validación de columna requerida
if "spike_anticipado" not in df.columns:
    print("[❌ ERROR] Falta columna requerida: spike_anticipado")
    exit()

# Separar clases
df_spike = df[df["spike_anticipado"] == 1]
df_no_spike = df[df["spike_anticipado"] == 0]

print(f"✅ Spike anticipado = 1: {len(df_spike)} filas")
print(f"✅ Spike anticipado = 0: {len(df_no_spike)} filas")

# Replicar clase minoritaria
df_spike_oversampled = pd.concat([df_spike] * 10, ignore_index=True)  # Aumentamos x10

# Combinar y mezclar
df_balanceado = pd.concat([df_spike_oversampled, df_no_spike], ignore_index=True)
df_balanceado = shuffle(df_balanceado, random_state=42)

# Reescribir el archivo original
df_balanceado.to_csv(RUTA_CSV, index=False)
print(f"✅ Dataset balanceado reescrito correctamente en: {RUTA_CSV}")
print(f"🧠 Total de filas tras balanceo: {len(df_balanceado)}")
