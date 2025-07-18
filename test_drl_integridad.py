# test_drl_integridad.py

import pandas as pd
from core.contexto import construir_contexto
from core.ia_drl import procesar_decision_drl
from utils.logs import log

CSV = "data/contexto_historico.csv"

def test_drl():
    log("[🧪 TEST DRL] Iniciando test de integridad DRL...")

    try:
        df = pd.read_csv(CSV)
        if len(df) < 60:
            log(f"[❌ TEST DRL] El CSV tiene menos de 60 filas: {len(df)}")
            return

        df = df.tail(120)
        contexto = construir_contexto(df, cantidad=60)

        if contexto is None:
            log("[❌ TEST DRL] El contexto construido es None.")
            return

        log("[✅ TEST DRL] Contexto construido correctamente:")
        for k, v in contexto.items():
            print(f" - {k}: {v}")

        capital = 500.0
        multiplicadores = [100, 200, 300, 400]
        decision = procesar_decision_drl(contexto, capital, multiplicadores)

        log("[🧪 RESULTADO DRL]")
        for k, v in decision.items():
            print(f" - {k}: {v}")

        if decision["monto"] <= 0 or decision["multiplicador"] <= 0 or decision["ganancia_esperada"] <= 0:
            log("[❌ TEST DRL] DRL retornó valores nulos. Hay un fallo interno.")
        else:
            log("[✅ TEST DRL] DRL respondió con valores válidos. Todo en orden.")

    except Exception as e:
        log(f"[❌ EXCEPCIÓN TEST DRL] {e}")

if __name__ == "__main__":
    test_drl()
