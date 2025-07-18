# test_diagnostico_contexto_spike.py

import pandas as pd
from core.contexto import construir_contexto_para_spike

CSV = "data/contexto_historico.csv"

def test_contexto_spike():
    try:
        df = pd.read_csv(CSV)
        if len(df) < 120:
            print(f"[❌ TEST] CSV con pocas velas: {len(df)}")
            return

        df_ultimas = df.tail(120).reset_index(drop=True)
        df_spike = construir_contexto_para_spike(df_ultimas)

        if df_spike is None:
            print("[❌ TEST] No se pudo construir contexto para Spike.")
        elif len(df_spike) < 60:
            print(f"[⚠️ TEST] Solo {len(df_spike)} filas válidas. Se esperaban 60.")
            print(df_spike)
        else:
            print(f"[✅ TEST PASADO] {len(df_spike)} filas válidas para Spike IA.")
            print(df_spike.tail(3))  # muestra las últimas 3 filas como ejemplo

    except Exception as e:
        print(f"[❌ ERROR TEST] {e}")

if __name__ == "__main__":
    test_contexto_spike()
