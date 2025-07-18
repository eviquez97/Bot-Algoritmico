import pandas as pd
import os

# Crear carpeta si no existe
os.makedirs("data", exist_ok=True)

# Datos de ejemplo válidos
datos = [
    {"open": 18540.0, "high": 18541.2, "low": 18538.9, "close": 18539.0, "spread": 2.3, "momentum": 0.6, "ema": 18539.5, "rsi": 70.5, "cierre_bueno": 1},
    {"open": 18539.0, "high": 18540.1, "low": 18537.5, "close": 18538.2, "spread": 2.6, "momentum": -0.4, "ema": 18538.8, "rsi": 65.3, "cierre_bueno": 0},
    {"open": 18538.2, "high": 18539.3, "low": 18536.7, "close": 18537.0, "spread": 2.6, "momentum": -0.7, "ema": 18537.9, "rsi": 61.1, "cierre_bueno": 1},
    {"open": 18537.0, "high": 18537.9, "low": 18535.6, "close": 18536.2, "spread": 2.3, "momentum": -0.8, "ema": 18536.9, "rsi": 58.0, "cierre_bueno": 0},
    {"open": 18536.2, "high": 18537.5, "low": 18534.8, "close": 18535.1, "spread": 2.7, "momentum": -1.1, "ema": 18535.8, "rsi": 54.2, "cierre_bueno": 1},
]

# Crear DataFrame y guardar CSV
df = pd.DataFrame(datos)
df.to_csv("data/dataset_cierre.csv", index=False)
print("✅ dataset_cierre.csv generado correctamente en carpeta /data")
