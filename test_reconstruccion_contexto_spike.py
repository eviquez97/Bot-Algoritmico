import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from core.contexto import construir_contexto_para_spike

RUTA_CSV = "data/contexto_historico.csv"
RUTA_MODELO_VISUAL = "modelos/scs_vision_x_model.keras"

# Paso 1: Cargar dataset
print("🔍 Cargando CSV de contexto...")
df = pd.read_csv(RUTA_CSV)

# Paso 2: Asegurar suficientes filas
if df.shape[0] < 120:
    print(f"❌ No hay suficientes filas en el CSV. Se requieren al menos 120 y hay {df.shape[0]}.")
    exit()

# Paso 3: Construir contexto desde las últimas 120 velas
df_contexto = construir_contexto_para_spike(df.tail(120))
if df_contexto is None or df_contexto.shape[0] < 30:
    print("❌ Falló la construcción del contexto Spike IA.")
    exit()

print(f"\n✅ Contexto construido: {df_contexto.shape[0]} filas, {df_contexto.shape[1]} columnas.")
print("🧬 Muestra del contexto final:")
print(df_contexto.tail(3))

# Paso 4: Preprocesar para la red visual CNN
X = df_contexto.tail(1).values.astype(np.float32)
X_scaled = StandardScaler().fit_transform(X)
X_scaled = np.expand_dims(X_scaled, axis=0)  # (1, 1, 9)

# Paso 5: Cargar modelo y predecir
print("\n🔬 Cargando modelo visual CNN...")
modelo = load_model(RUTA_MODELO_VISUAL)

pred = modelo.predict(X_scaled, verbose=0)[0][0]
print(f"\n🧠 Predicción Visual CNN (spike anticipado): {pred:.4f}")
