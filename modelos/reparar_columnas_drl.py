import pandas as pd

ruta_dataset = "data/dataset_drl.csv"

df = pd.read_csv(ruta_dataset)

# Renombrar columnas
renombrar = {
    "pred_futuro": "futuro",
    "porcentaje_bajistas": "bajistas",
    "pred_visual": "visual_spike",
    "pred_rf": "rf_spike",
    "pred_lstm": "lstm_spike",
}

df.rename(columns=renombrar, inplace=True)

# Crear columna ema_diff
if "ema" in df.columns:
    df["ema_diff"] = df["ema"].diff().fillna(0)

# Guardar dataset reparado
df.to_csv(ruta_dataset, index=False)

print("✅ Dataset DRL reparado: columnas renombradas y 'ema_diff' generada.")
print(f"📄 Columnas actuales: {df.columns.tolist()}")
