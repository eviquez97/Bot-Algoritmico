# test_escritura_csv.py

import csv
import os
import time

ARCHIVO = "data/contexto_historico.csv"  # o cambia a dataset/dataset_operativo.csv
CANTIDAD_COLUMNAS_ESPERADAS = 16  # o 17 si ya incluiste spike_real, anticipado, etc.

def contar_filas_invalidas():
    if not os.path.exists(ARCHIVO):
        print("❌ Archivo no encontrado:", ARCHIVO)
        return

    with open(ARCHIVO, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        fila_numero = 2  # porque el header fue la fila 1

        for fila in reader:
            columnas = len(fila)
            if columnas != len(header):
                print(f"\n⚠️ Fila corrupta detectada en línea {fila_numero}")
                print(f"📏 Columnas esperadas: {len(header)} | Encontradas: {columnas}")
                print(f"🧾 Contenido de la fila: {fila}")
                return fila_numero, columnas, fila
            fila_numero += 1

    print("✅ Todas las filas tienen el número correcto de columnas.")

# Bucle que monitorea el archivo cada 10 segundos
print(f"🧪 Iniciando monitoreo del archivo: {ARCHIVO}")
while True:
    contar_filas_invalidas()
    time.sleep(10)  # espera 10 segundos antes de revisar de nuevo
