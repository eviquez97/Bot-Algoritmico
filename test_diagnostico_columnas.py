# test_diagnostico_columnas.py

import os
import pandas as pd

# Directorio donde están tus CSV
carpeta_csv = "data"

# Columnas que comúnmente causan errores
columnas_criticas = ['target', 'score', 'accion', 'ganancia_estimada']

# Recorre todos los archivos CSV
for archivo in os.listdir(carpeta_csv):
    if archivo.endswith(".csv"):
        ruta = os.path.join(carpeta_csv, archivo)
        print(f"\n📄 Analizando archivo: {ruta}")

        try:
            df = pd.read_csv(ruta)
            columnas = df.columns.tolist()
            print(f"✅ Columnas detectadas: {columnas}")

            for col in columnas_criticas:
                if col not in columnas:
                    print(f"⚠️  FALTA columna crítica: '{col}' en {archivo}")

        except Exception as e:
            print(f"❌ ERROR al leer {archivo}: {e}")
