import pandas as pd

# Ruta original
RUTA_ORIGINAL = 'data/contexto_historico.csv'
RUTA_SALIDA = 'data/contexto_historico.csv'  # ¡Reemplaza el mismo archivo!

# Cargar CSV
df = pd.read_csv(RUTA_ORIGINAL)

# Verificar columnas necesarias
if 'close' not in df.columns:
    raise ValueError("La columna 'close' es obligatoria para calcular la ganancia futura.")

# Agregar columna de ganancia futura (cierre siguiente vela - cierre actual)
df['ganancia_futura'] = df['close'].shift(-1) - df['close']

# Eliminar la última fila (donde no hay ganancia futura conocida)
df = df.dropna(subset=['ganancia_futura'])

# Guardar sobreescribiendo el original
df.to_csv(RUTA_SALIDA, index=False)

print(f"✅ Columna 'ganancia_futura' añadida correctamente a {RUTA_SALIDA}")
