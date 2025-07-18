import pandas as pd
import shutil
import os

RUTA_ORIGINAL = "data/dataset_spike_monstruo_limpio.csv"
RUTA_BACKUP = "data/dataset_spike_monstruo_limpio_backup.csv"

try:
    # Verificar existencia
    if not os.path.exists(RUTA_ORIGINAL):
        raise FileNotFoundError("❌ No se encontró el archivo original.")

    # Crear backup de seguridad
    shutil.copy(RUTA_ORIGINAL, RUTA_BACKUP)
    print(f"🛡️ Backup creado en: {RUTA_BACKUP}")

    # Cargar dataset original
    df = pd.read_csv(RUTA_ORIGINAL)

    if "spike" not in df.columns:
        raise ValueError("❌ El archivo no contiene una columna 'spike' necesaria para generar 'spike_anticipado'.")

    # Crear columna spike_anticipado (valor de la fila siguiente)
    df["spike_anticipado"] = df["spike"].shift(-1)

    # Rellenar última fila vacía con 0 (no hay fila siguiente)
    df["spike_anticipado"].fillna(0, inplace=True)

    # Asegurar que sea tipo entero
    df["spike_anticipado"] = df["spike_anticipado"].astype(int)

    # Guardar sobreescribiendo el original
    df.to_csv(RUTA_ORIGINAL, index=False)

    print("✅ Columna 'spike_anticipado' añadida correctamente al archivo original.")

except Exception as e:
    print(f"❌ Error al procesar: {e}")
