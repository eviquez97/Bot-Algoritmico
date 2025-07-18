# core/drl.py

from core.validaciones import filtro_entrada_ultra_ia
from core.spike_guardian_predictivo import evaluar_spike_anticipado
from core.drl_estado import obtener_estado_drl
from core.smart_compound import obtener_entrada_dinamica
from core.watchdog import contrato_activo
from core.websocket_orden import enviar_orden_ws
from core.drl_agente import drl_agent
from utils.logs import log
from config import SYMBOL
from datetime import datetime

def procesar_decision_drl(contexto, df_actual):
    try:
        if contrato_activo():
            log("[⏸️ DRL DESCARTADO] Ya hay un contrato activo.")
            return

        momentum = df_actual["momentum"].iloc[0]
        if momentum >= 0:
            log("[❌ DRL DESCARTADO] Momentum alcista.")
            return

        if not filtro_entrada_ultra_ia(contexto, df_actual):
            log("[❌ DRL DESCARTADO] Filtro Ultra IA bloqueó la entrada.")
            return

        if evaluar_spike_anticipado(contexto):
            log("[🚫 DRL BLOQUEADO] Spike anticipado por IA predictiva.")
            return

        # Obtener estado del mercado y acción sugerida por DRL
        estado = obtener_estado_drl(df_actual, contexto)
        accion_idx = drl_agent.actuar(estado)

        # Obtener score y predicción futura para entrada inteligente
        score = df_actual["score"].iloc[0]
        pred_futuro = df_actual["prediccion_futura"].iloc[0]
        monto_final, multiplicador = obtener_entrada_dinamica(score, pred_futuro)

        # Guardar en contexto para seguimiento
        contexto["estado_vector"] = estado
        contexto["accion_idx"] = accion_idx
        contexto["precio_entrada"] = df_actual["close"].iloc[0]
        contexto["symbol"] = SYMBOL
        contexto["timestamp"] = datetime.utcnow().isoformat()

        # Enviar orden real
        enviar_orden_ws(monto_final, multiplicador, contexto)

    except Exception as e:
        log(f"[❌ ERROR DECISIÓN DRL] {type(e).__name__} - {e}", "error")

