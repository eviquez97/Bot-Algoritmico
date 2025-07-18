# core/registro_ganancia.py

import pandas as pd
import os
from utils.logs import log

CSV_CONTEXTO = "data/contexto_historico.csv"

def registrar_ganancia_real(generada):
    """
    Actualiza la última fila del CSV con la ganancia obtenida tras el cierre del contrato.
    """
    try:
        if not os.path.exists(CSV_CONTEXTO):
            log("[❌ REGISTRO GANANCIA] No existe el CSV para actualizar.")
            return

        df = pd.read_csv(CSV_CONTEXTO)
        if df.empty:
            log("[❌ REGISTRO GANANCIA] El CSV está vacío.")
            return

        df.at[df.index[-1], "ganancia_estim"] = generada

        df.to_csv(CSV_CONTEXTO, index=False)
        log(f"[💰 GANANCIA REGISTRADA] Última fila actualizada con ganancia: ${generada:.2f}")

    except Exception as e:
        log(f"[❌ ERROR REGISTRO GANANCIA] {e}")
