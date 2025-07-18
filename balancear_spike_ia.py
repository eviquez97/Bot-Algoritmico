import pandas as pd
from sklearn.utils import resample

# Ruta del archivo original
ruta = "data/dataset_spike_monstruo_limpio.csv"

# Cargar CSV
df = pd.read_csv(ruta)

# Eliminar filas con spike_anticipado nulo
df = df.dropna(subset=["spike_anticipado"])

# Asegurar tipo entero en spike_anticipado
df["spike_anticipado"] = df["spike_anticipado"].astype(int)

# Separar clases
df_spike = df[df["spike_anticipado"] == 1]
df_no_spike = df[df["spike_anticipado"] == 0]

# Aplicar undersampling a la clase mayoritaria
df_no_spike_downsampled = resample(
    df_no_spike,
    replace=False,
    n_samples=len(df_spike),
    random_state=42
)

# Unir clases balanceadas
df_balanceado = pd.concat([df_spike, df_no_spike_downsampled]).sample(frac=1, random_state=42)

# Sobrescribir el mismo archivo
df_balanceado.to_csv(ruta, index=False)

# Confirmación
print(f"✅ Dataset sobrescrito: {ruta}")
print(f"📊 Clases finales: {df_balanceado['spike_anticipado'].value_counts().to_dict()}")
