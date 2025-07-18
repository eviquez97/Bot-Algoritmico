import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Ruta dataset
RUTA = "data/dataset_spike_monstruo_limpio.csv"

# Columnas de entrada y objetivo
columnas = ['fuerza_cuerpo', 'fuerza_mecha', 'bajistas', 'rsi', 'momentum', 'spread', 'score', 'ema', 'variacion']
objetivo = "spike_anticipado"

# Cargar dataset
df = pd.read_csv(RUTA)
df = df[columnas + [objetivo]].dropna()

X = df[columnas]
y = df[objetivo]

# Entrenar modelo
model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
model.fit(X, y)

# Guardar modelo limpio
with open("modelos/model_spike.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Modelo RandomForest (spike anticipado) reentrenado y guardado correctamente como model_spike.pkl")
