import pandas as pd

# Ruta al archivo CSV
RUTA_CSV = "data/contexto_historico.csv"

# Cargar el archivo
df = pd.read_csv(RUTA_CSV)

# Eliminar filas vacías o corruptas
df.dropna(inplace=True)

# Asegurar que las columnas necesarias existan
if not {'open', 'close'}.issubset(df.columns):
    print("❌ El archivo no contiene columnas 'open' y 'close'.")
    exit()

# Detectar velas verdes (spike real)
spikes = df[df['close'] > df['open']]

print(f"\n🔍 Se encontraron {len(spikes)} velas verdes (spikes reales):\n")

for i, row in spikes.iterrows():
    print(f"🟢 Spike en fila {i} | Open: {row['open']} | Close: {row['close']} | Variación: {row['close'] - row['open']:.4f}")

print("\n✅ Análisis completado.\n")
