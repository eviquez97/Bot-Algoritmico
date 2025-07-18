import pandas as pd

CSV_PATH = "data/dataset_operativo.csv"

# Columnas oficiales
COLUMNAS = [
    "timestamp", "fecha", "hora",
    "open", "high", "low", "close",
    "spread", "fuerza_cuerpo", "fuerza_mecha", "score",
    "rsi", "ema", "momentum", "variacion",
    "alcista", "volumen_tick",
    "spike_real", "spike_anticipado",
    "ganancia_esperada", "duracion_estimada", "operacion_exitosa"
]

# Leer sin encabezado
with open(CSV_PATH, "r") as f:
    lineas = f.readlines()

lineas_limpias = []
for i, linea in enumerate(lineas):
    columnas = linea.strip().split(",")
    if len(columnas) == 22:
        lineas_limpias.append(linea)
    else:
        print(f"[⚠️ FILA CORRUPTA] Línea {i+1} eliminada por tener {len(columnas)} columnas.")

# Guardar CSV limpio
with open(CSV_PATH, "w") as f:
    f.write(",".join(COLUMNAS) + "\n")
    f.writelines(lineas_limpias)

print(f"[✅ REPARACIÓN COMPLETA] Se guardó archivo limpio con {len(lineas_limpias)} filas válidas.")
