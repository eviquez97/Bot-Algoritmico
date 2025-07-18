import pandas as pd

columnas = [
    "score", "ganancia_esperada", "bajistas", "futuro", "ema", "rsi", "momentum", "spread",
    "rf_spike", "lstm_spike", "visual_spike", "ultima_direccion",
    "accion", "monto", "multiplicador", "resultado", "exito"
]

df = pd.DataFrame(columns=columnas)
df.to_csv("data/dataset_drl.csv", index=False)
print("✅ Dataset DRL limpio generado correctamente.")
