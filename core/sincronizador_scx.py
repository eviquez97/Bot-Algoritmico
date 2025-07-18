# core/sincronizador_scx.py

from core.buffer import VELAS_BUFFER
from core.ia_cierre import evaluar_scpx
from core.operaciones import contrato_activo
from core.objetivo_diario import registrar_ganancia
from utils.logs import log

# Esta función se llama cada vez que llega una vela
def evaluar_cierre_predictivo_sincronizado():
    if not contrato_activo:
        return  # No hay contrato que evaluar

    if len(VELAS_BUFFER) < 30:
        return

    try:
        vela_actual = VELAS_BUFFER[-1]
        df_actual = pd.DataFrame([vela_actual])

        resultado = evaluar_scpx(df_actual)

        if resultado["cerrar"]:
            log(f"[⚠️ CIERRE SCM+SCDP-X] Señal de cierre detectada | Motivo: {resultado['motivo']}")
            registrar_ganancia(resultado.get("ganancia", 0))  # Registrar ganancia si aplica
            # ⚠️ Aquí deberías invocar tu lógica de cierre real del contrato
            # cerrar_contrato_activo()
    except Exception as e:
        log(f"[❌ ERROR SINCRO SCM↔SCDP-X] {e}")
