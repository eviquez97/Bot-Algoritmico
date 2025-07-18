import pandas as pd
import joblib

from core.contexto import construir_contexto_para_spike

# Rutas
RUTA_CSV = "data/contexto_historico.csv"
RUTA_MODELO = "modelos/model_spike.pkl"

# Cargar CSV completo
df_csv = pd.read_csv(RUTA_CSV)
print(f"🔍 CSV cargado: {len(df_csv)} filas.")

# Tomar las últimas 120 velas
df_spike = df_csv.tail(120)
print(f"\n📊 Últimas 120 filas para construir contexto:\n{df_spike.tail(5)}")

# Construir contexto
contexto = construir_contexto_para_spike(df_spike)

if contexto is None or contexto.empty:
    print("\n❌ Error: No se pudo construir el contexto.")
else:
    print(f"\n✅ Contexto construido correctamente: {contexto.shape[0]} filas, {contexto.shape[1]} columnas.")
    print(f"\n🧬 Muestra del contexto final que se le pasa al modelo:")
    print(contexto.tail(3))

    # Cargar modelo
    modelo = joblib.load(RUTA_MODELO)

    # Realizar predicción
    probas = modelo.predict_proba(contexto)[-1][1]  # Tomamos la probabilidad de la última fila
    print(f"\n🧠 Predicción RF (spike anticipado) última fila: {probas:.4f}")
