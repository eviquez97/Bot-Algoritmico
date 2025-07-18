from utils.logs import log

def filtro_entrada_ultra_ia(contexto, df_actual):
    try:
        score = df_actual["score"].iloc[0]
        bajistas = df_actual["porcentaje_bajistas"].iloc[0]
        ganancia_estim = df_actual["ganancia_estimada"].iloc[0]

        if score < 0.75:
            log("[❌ FILTRO ULTRA IA] Score bajo.")
            return False

        if bajistas < 75:
            log("[❌ FILTRO ULTRA IA] Porcentaje bajistas insuficiente.")
            return False

        if ganancia_estim < 20:
            log("[❌ FILTRO ULTRA IA] Ganancia estimada insuficiente.")
            return False

        return True

    except Exception as e:
        log(f"[⚠️ ERROR FILTRO ULTRA IA] {type(e).__name__} - {e}", "warning")
        return False
