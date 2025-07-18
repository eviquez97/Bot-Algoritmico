import os
import sys
import time
from datetime import datetime
from utils.logs import log
from core.buffer import VELAS_BUFFER

estado_sistema = {
    "ultimo_tick": time.time(),
    "ultimo_contrato": time.time(),
    "reinicios": 0
}

def marcar_tick_recibido():
    estado_sistema["ultimo_tick"] = time.time()

def marcar_contrato_ejecutado():
    estado_sistema["ultimo_contrato"] = time.time()

def verificar_autocuracion():
    ahora = time.time()
    delta_tick = ahora - estado_sistema["ultimo_tick"]
    delta_contrato = ahora - estado_sistema["ultimo_contrato"]

    # 🔕 AUTOCURACIÓN DESACTIVADA TEMPORALMENTE PARA EVITAR REINICIOS
    log("[🧪 AUTOCURACIÓN] 🔕 Modo seguro activado: No se forzarán reinicios automáticos.")

    # Si quieres volver a activar los reinicios, comenta la línea de arriba
    # y descomenta el siguiente bloque:

    """
    if len(VELAS_BUFFER) == 0:
        log("[🧪 AUTOCURACIÓN] Buffer vacío. Reiniciando proceso por seguridad...")
        reiniciar_bot()
    elif delta_tick > 120:
        log(f"[🧪 AUTOCURACIÓN] No se reciben ticks desde hace {int(delta_tick)}s. Reiniciando bot...")
        reiniciar_bot()
    elif delta_contrato > 600:
        log(f"[🧪 AUTOCURACIÓN] No se ejecutan contratos desde hace {int(delta_contrato/60)} minutos. Reiniciando bot...")
        reiniciar_bot()
    """

def reiniciar_bot():
    estado_sistema["reinicios"] += 1
    log(f"[🔄 REINICIO] Reiniciando bot automáticamente. Total reinicios: {estado_sistema['reinicios']}")
    os.execl(sys.executable, sys.executable, *sys.argv)

