import os
import pandas as pd

# Crear carpeta si no existe
os.makedirs("data", exist_ok=True)

# Estructura de columnas esperadas
columnas_drl = [
    "score", "futuro", "bajistas",
    "visual_spike", "rf_spike", "lstm_spike",
    "ema_diff", "rsi", "momentum", "spread",
    "monto", "multiplicador"
]

# Crear un DataFrame vacío con esas columnas
df = pd.DataFrame(columns=columnas_drl)

# Guardarlo como CSV
df.to_csv("data/dataset_drl.csv", index=False)
print("✅ dataset_drl.csv creado con 12 columnas válidas.")
