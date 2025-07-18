# core/filtros.py

from utils.logs import log

def es_entrada_necesaria(df_actual, porcentaje_bajistas, pred_futuro):
    """
    Evalúa si las condiciones actuales permiten una entrada.
    Retorna True si se permite entrar, False en caso contrario.
    """

    try:
        if df_actual.empty:
            return False

        # Permitir entrada siempre que se llame desde DRL válido
        return True

    except Exception as e:
        log(f"[❌ ERROR FILTRO ENTRADA] {e}")
        return False
