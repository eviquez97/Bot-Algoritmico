# core/recompensa_drl.py

from utils.logs import log
import numpy as np

def registrar_recompensa_drl(
    contexto,
    ganancia=None,
    estado_vector_manual=None,
    accion_manual=None,
    recompensa_manual=None,
):
    try:
        if (
            recompensa_manual is not None
            and estado_vector_manual is not None
            and accion_manual is not None
        ):
            recompensa = recompensa_manual
            estado = estado_vector_manual
            accion = accion_manual
            siguiente_estado = np.zeros((1, 10), dtype=np.float32)
            terminado = True
        else:
            recompensa = 0
            if ganancia is None:
                log("[⚠️ DRL] No se recibió ganancia para calcular recompensa.")
                return

            if ganancia >= contexto.get("target_profit", 0.30):
                recompensa = 1.0
            elif ganancia >= 0.15:
                recompensa = 0.5
            elif ganancia < -1.0:
                recompensa = -1.0
            else:
                recompensa = -0.5

            estado = contexto.get("estado_vector", np.zeros((1, 10)))
            accion = contexto.get("accion_idx", 0)
            siguiente_estado = estado
            terminado = True

        from core.drl import drl_agent

        drl_agent.guardar(estado, accion, recompensa, siguiente_estado, terminado)
        drl_agent.entrenar()
        drl_agent.guardar_modelo()

        log(f"[🧠 DRL ENTRENADO] Acción {accion} | Recompensa: {recompensa}")
        print(f"🧮 Acción DRL → ID: {accion}")
        print(f"🔁 DRL Exploración (epsilon): {drl_agent.epsilon:.3f}")

    except Exception as e:
        log(f"[❌ ERROR EN RECOMPENSA DRL] {e}", "error")
