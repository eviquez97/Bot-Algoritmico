import csv

RUTA = "data/contexto_historico.csv"
COLUMNAS_ESP = 16
filas_validas = []

# Leer y conservar solo las filas correctas
with open(RUTA, "r") as f:
    lector = csv.reader(f)
    for fila in lector:
        if len(fila) == COLUMNAS_ESP:
            filas_validas.append(fila)

# Sobrescribir el mismo archivo con las filas válidas
with open(RUTA, "w", newline="") as f:
    escritor = csv.writer(f)
    escritor.writerows(filas_validas)

print(f"✅ Archivo corregido en el mismo lugar.")
print(f"✔️ Filas válidas conservadas: {len(filas_validas)}")
