# core/decision_drl.py

import numpy as np
from utils.logs import log
from config import SYMBOL
from core.websocket_orden import enviar_orden_ws
from core.smart_compound import compound_manager
from core.drl import drl_agent
from core.drl_estado import obtener_estado_drl
from datetime import datetime

def decidir_y_ejecutar(contexto, df_actual):
    try:
        estado = obtener_estado_drl(df_actual, contexto)
        accion_idx = drl_agent.actuar(estado)

        monto_final, multiplicador = compound_manager.obtener_entrada_dinamica(accion_idx)

        contexto["estado_vector"] = estado
        contexto["accion_idx"] = accion_idx
        contexto["precio_entrada"] = df_actual["close"].iloc[0]
        contexto["symbol"] = SYMBOL
        contexto["timestamp"] = datetime.utcnow().isoformat()

        log(f"[🚀 DRL] Ejecutando orden | Monto: ${monto_final:.2f} | Multiplicador: x{multiplicador}")
        enviar_orden_ws(monto_final, multiplicador, contexto)

    except Exception as e:
        log(f"[❌ ERROR DECISIÓN DRL] {e}", "error")
