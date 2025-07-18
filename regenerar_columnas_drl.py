import pandas as pd
import joblib

df = pd.read_csv("data/dataset_operativo.csv")

columnas_a_excluir = [
    "timestamp", "fecha", "hora", "spike_real", "spike_anticipado",
    "ganancia_esperada", "duracion_estimada", "operacion_exitosa", "alcista", "bajistas"
]

columnas_drl = [col for col in df.columns if col not in columnas_a_excluir]

joblib.dump(columnas_drl, "modelos/columnas_drl.pkl")

print("✅ columnas_drl.pkl regenerado correctamente.")
print(f"📦 Total columnas usadas para DRL: {len(columnas_drl)}")
