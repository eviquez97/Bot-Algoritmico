import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Crear 300 velas simuladas de 1 minuto
n = 300
inicio = datetime.now()

datos = []
precio = 1000.0

for i in range(n):
    ts = int((inicio + timedelta(minutes=i)).timestamp())
    open_ = precio
    high = open_ + np.random.uniform(0.5, 2.0)
    low = open_ - np.random.uniform(0.5, 2.0)
    close = np.random.uniform(low, high)
    datos.append({
        "epoch": ts,
        "open": round(open_, 5),
        "high": round(high, 5),
        "low": round(low, 5),
        "close": round(close, 5)
    })
    precio = close  # siguiente vela parte del cierre actual

df = pd.DataFrame(datos)

# Crear carpeta si no existe
os.makedirs("data", exist_ok=True)
df.to_csv("data/velas_1m.csv", index=False)
print("✅ Archivo de ejemplo guardado en data/velas_1m.csv")
