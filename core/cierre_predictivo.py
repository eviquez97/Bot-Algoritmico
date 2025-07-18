# core/cierre_predictivo.py

from core.ia_cierre import evaluar_scpx
from core.smart_compound import evaluar_scm
from core.ia_spike import evaluar_spike_ia
from core.estado import contrato_activo, datos_operacion
from core.operaciones import cerrar_contrato_activo
from utils.logs import log

def cierre_spike_predictivo(df_actual):
    try:
        if contrato_activo is None:
            log("[📉 CIERRE OMITIDO] No hay contrato activo para evaluar cierre.")
            return

        # 🔍 Evaluación de Spike IA
        resultado_spike = evaluar_spike_ia(df_actual)
        if resultado_spike.get("bloqueado", False):
            log("💣 CIERRE INTELIGENTE ACTIVADO | Motivo: Spike anticipado por Spike IA")
            cerrar_contrato_activo()
            return

        # 🔍 Evaluación IA Cierre (SCDP-X) y Smart Compound
        cerrar_por_scp = evaluar_scpx(df_actual)
        cerrar_por_scm = evaluar_scm(df_actual)

        if cerrar_por_scp:
            log("💣 CIERRE INTELIGENTE ACTIVADO | Motivo: Spike detectado por SCDP-X")
            cerrar_contrato_activo()
            return

        if cerrar_por_scm:
            log("💣 CIERRE INTELIGENTE ACTIVADO | Motivo: Evaluación SCM cumplida")
            cerrar_contrato_activo()
            return

        # 💰 Cierre por ganancia alcanzada
        ganancia_real = datos_operacion.get("ganancia_real", 0.0)
        ganancia_esperada = datos_operacion.get("ganancia_esperada", 9999.0)

        if ganancia_real >= ganancia_esperada:
            log(f"💰 CIERRE ACTIVADO POR GANANCIA | Real: ${ganancia_real:.2f} >= Esperada: ${ganancia_esperada:.2f}")
            cerrar_contrato_activo()
        else:
            log(f"🔍 [CIERRE] Evaluado, aún sin condiciones para cerrar. Ganancia actual: ${ganancia_real:.2f} / {ganancia_esperada:.2f}")

    except Exception as e:
        log(f"[❌ ERROR CIERRE IA] {e}")
