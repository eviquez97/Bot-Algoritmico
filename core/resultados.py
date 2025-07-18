# core/resultados.py

from core.smart_compound import actualizar_scm
from core.meta_diaria import registrar_ganancia
from utils.logs import log

def registrar_resultado_contrato(resultado):
    try:
        actualizar_scm(resultado)
        registrar_ganancia(resultado)
        log(f"[📊 RESULTADO CONTRATO] Resultado registrado: {resultado:.2f}")
    except Exception as e:
        log(f"[❌ ERROR REGISTRO RESULTADO] {e}")
