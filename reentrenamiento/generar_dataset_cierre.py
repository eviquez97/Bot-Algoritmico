import pandas as pd
import numpy as np
import os

# Crear dataset simulado
np.random.seed(42)
filas = 300

data = {
    "open": np.random.uniform(10000, 20000, filas),
    "high": np.random.uniform(10000, 20000, filas),
    "low": np.random.uniform(10000, 20000, filas),
    "close": np.random.uniform(10000, 20000, filas),
    "spread": np.random.uniform(0.5, 3.0, filas),
    "ema": np.random.uniform(10000, 20000, filas),
    "rsi": np.random.uniform(10, 90, filas),
    "momentum": np.random.uniform(-5, 5, filas),
    "resultado": np.random.choice([0, 1], filas, p=[0.6, 0.4])  # 0 = no cerrar, 1 = cerrar
}

df = pd.DataFrame(data)

# Asegurarse que la carpeta exista
os.makedirs("data", exist_ok=True)

# Guardar archivo
df.to_csv("data/dataset_cierre.csv", index=False)
print("✅ Dataset simulado de cierre guardado en data/dataset_cierre.csv")
