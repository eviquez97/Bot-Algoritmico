import pandas as pd
from core.contexto import construir_contexto
from core.ia_drl import procesar_decision_drl
from utils.logs import log

CSV_PATH = "data/contexto_historico.csv"

def test_drl_predict():
    try:
        df = pd.read_csv(CSV_PATH)

        if len(df) < 60:
            log(f"[❌ TEST DRL] No hay suficientes velas en el CSV. Requiere al menos 60, hay {len(df)}")
            return

        df = df.tail(120)  # contexto crudo
        contexto = construir_contexto(df, cantidad=60)

        if contexto is None:
            log("[❌ TEST DRL] Error al construir contexto. Retorno = None")
            return

        capital_simulado = 500.0
        multiplicadores = [100, 200, 300, 400]

        log("🔍 Ejecutando DRL con contexto actual...")

        decision = procesar_decision_drl(contexto, capital_simulado, multiplicadores)

        print("\n================= RESULTADO DRL =================")
        print(f"Permitir entrada : {decision['permitir_entrada']}")
        print(f"Score DRL        : {decision['score']:.4f}")
        print(f"Acción elegida   : {decision['accion']}")
        print(f"Monto            : ${decision['monto']:.2f}")
        print(f"Multiplicador    : {decision['multiplicador']}")
        print(f"Ganancia esperada: ${decision['ganancia_esperada']:.2f}")
        print(f"Duración estimada: {decision['duracion_estimada']}s")
        print("================================================\n")

    except Exception as e:
        log(f"[❌ ERROR TEST DRL] {e}")

if __name__ == "__main__":
    test_drl_predict()
