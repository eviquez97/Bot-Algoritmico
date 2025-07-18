import pandas as pd
from core.ia_spike import evaluar_spike_ia

CSV_PATH = "data/contexto_historico.csv"

def detectar_ultimo_spike(df):
    """
    Retorna el índice de la última vela verde explosiva (close > open).
    """
    for i in range(len(df) - 1, -1, -1):
        if df.iloc[i]["close"] > df.iloc[i]["open"]:
            return i
    return None

def main():
    try:
        df = pd.read_csv(CSV_PATH).dropna()
        if len(df) < 120:
            print("❌ No hay suficientes velas para el test.")
            return

        idx_spike = detectar_ultimo_spike(df)
        if idx_spike is None or idx_spike < 60:
            print("❌ No se encontró un spike reciente válido.")
            return

        contexto_previo = df.iloc[idx_spike - 60:idx_spike]

        print(f"\n🧠 Evaluando Spike IA sobre las 60 velas previas al spike (índice: {idx_spike})...")
        resultado = evaluar_spike_ia(contexto_previo)

        rf = resultado.get("rf", 0)
        lstm = resultado.get("lstm", 0)
        visual = resultado.get("visual", 0)
        bloqueado = resultado.get("bloqueado", False)

        print(f"🧠 SPIKE IA V5 (anticipación): RF={rf:.2f} | LSTM={lstm:.2f} | Visual={visual:.2f} | Bloqueado={bloqueado}")
        if bloqueado:
            print("✅ El sistema SÍ podría haber anticipado el spike.")
        else:
            print("⚠️ El sistema NO lo habría anticipado con los umbrales actuales.")

    except Exception as e:
        print(f"[❌ ERROR TEST] {e}")

if __name__ == "__main__":
    main()
