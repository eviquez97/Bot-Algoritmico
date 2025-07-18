RUTA = "data/dataset_drl.csv"

try:
    with open(RUTA, "r", encoding="utf-8") as f:
        lineas = f.readlines()

    encabezado = lineas[0].strip().split(",")
    columnas_esperadas = len(encabezado)

    lineas_validas = [lineas[0]]
    filas_corruptas = 0

    for i, linea in enumerate(lineas[1:], start=2):
        columnas = linea.strip().split(",")
        if len(columnas) == columnas_esperadas:
            lineas_validas.append(linea)
        else:
            filas_corruptas += 1
            print(f"[⚠️ LÍNEA INVALIDA] Línea {i}: {len(columnas)} columnas")

    with open(RUTA, "w", encoding="utf-8") as f:
        f.writelines(lineas_validas)

    print(f"\n✅ Reparación completada: {filas_corruptas} filas corruptas eliminadas.")
    print(f"📄 Total de filas válidas: {len(lineas_validas)-1}")
except Exception as e:
    print(f"❌ Error al reparar el dataset: {e}")
