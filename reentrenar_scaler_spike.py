import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Ruta del dataset
RUTA = "data/dataset_spike_monstruo_limpio.csv"

# Columnas que vamos a escalar (las que usan todos los modelos Spike IA)
columnas = ['fuerza_cuerpo', 'fuerza_mecha', 'bajistas', 'rsi', 'momentum', 'spread', 'score', 'ema', 'variacion']

# Cargar dataset
df = pd.read_csv(RUTA)

# Validar columnas
df = df[columnas].dropna()

# Entrenar scaler
scaler = StandardScaler()
scaler.fit(df)

# Guardar scaler
with open("modelos/scaler_spike.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Scaler reentrenado y guardado correctamente como scaler_spike.pkl")
