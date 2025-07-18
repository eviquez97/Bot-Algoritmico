import pandas as pd

df = pd.read_csv("data/dataset_drl.csv")

# Crear columnas Q0, Q1, Q2, Q3
for i in range(4):
    df[f"Q{i}"] = 0.0

# Asignar recompensa 1.0 a la acción correcta si fue exitosa, 0.0 si no
for i, row in df.iterrows():
    accion = int(row["accion"])
    exito = row["exito"]
    df.at[i, f"Q{accion}"] = 1.0 if exito else 0.0

# Guardar
df.to_csv("data/dataset_drl.csv", index=False)
print("✅ Dataset DRL reparado con columnas Q0–Q3 correctamente.")
