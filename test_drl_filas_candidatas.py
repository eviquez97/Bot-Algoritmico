import pandas as pd

CSV_PATH = "data/dataset_drl.csv"

def encontrar_filas_candidatas(csv_path):
    df = pd.read_csv(csv_path)

    print(f"✅ Dataset cargado: {len(df)} filas")

    # Limpiar filas con NaN
    df = df.dropna()

    # Filtro inteligente
    candidatos = df[
        (df["rf_spike"] > 0) |
        (df["lstm_spike"] > 0) |
        (df["visual_spike"] > 0)
    ]
    candidatos = candidatos[
        (candidatos["ganancia_estimada"] > 10) &
        (candidatos["score"] > 1) &
        (candidatos["rsi"] > 1) &
        (candidatos["momentum"].abs() > 0.5) &
        (candidatos["spread"] > 0.5)
    ]

    if candidatos.empty:
        print("⚠️ No se encontraron filas ideales para test DRL con spikes activos.")
    else:
        print(f"🔍 Filas candidatas encontradas: {len(candidatos)}")
        print("📌 Mostrando las 5 mejores (ordenadas por ganancia estimada):\n")
        top = candidatos.sort_values(by="ganancia_estimada", ascending=False).head(5)
        print(top[["timestamp", "ganancia_estimada", "score", "rsi", "momentum", "spread", "rf_spike", "lstm_spike", "visual_spike", "accion"]])

if __name__ == "__main__":
    encontrar_filas_candidatas(CSV_PATH)
