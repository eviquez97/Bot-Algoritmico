# limpiar_csv_spike_monstruo.py
import csv

archivo = "data/dataset_spike_monstruo_limpio.csv"
salida = "data/dataset_spike_monstruo_limpio.csv"  # Sobrescribe

try:
    with open(archivo, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        header = reader[0]
        columnas_esperadas = len(header)
        filas_validas = [header]

        for i, fila in enumerate(reader[1:], start=2):
            if len(fila) == columnas_esperadas:
                filas_validas.append(fila)
            else:
                print(f"[⚠️ FILA CORRUPTA] Línea {i}: columnas={len(fila)} (esperadas={columnas_esperadas})")

    with open(salida, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerows(filas_validas)

    print(f"[✅ ARCHIVO CORREGIDO] Guardado sin filas corruptas ({len(filas_validas)-1} válidas)")

except Exception as e:
    print(f"[❌ ERROR] {e}")
