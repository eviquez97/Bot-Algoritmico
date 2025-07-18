import csv

RUTA = "data/contexto_historico.csv"
COLUMNAS_ESPERADAS = 16

with open(RUTA, "r", newline="") as archivo_entrada:
    lineas_validas = []
    lector = csv.reader(archivo_entrada)
    encabezado = next(lector)
    for fila in lector:
        if len(fila) == COLUMNAS_ESPERADAS:
            lineas_validas.append(fila)

with open(RUTA, "w", newline="") as archivo_salida:
    escritor = csv.writer(archivo_salida)
    escritor.writerow(encabezado)
    escritor.writerows(lineas_validas)

print(f"[✅ LIMPIEZA] Se conservaron {len(lineas_validas)} filas con exactamente {COLUMNAS_ESPERADAS} columnas.")
