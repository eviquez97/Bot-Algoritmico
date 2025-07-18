# core/scdp_x.py

import numpy as np
from modelos.modelos_scdp import modelo_cierre, scaler_scdp
from core.buffer import VELAS_BUFFER
from utils.logs import log
from core.operaciones import contrato_activo
from core.estrategia_diaria import registrar_resultado_operacion

def evaluar_scpx():
    if not contrato_activo or len(VELAS_BUFFER) < 30:
        return

    try:
        df = VELAS_BUFFER[-30:]
        columnas = ['open', 'high', 'low', 'close', 'spread', 'ema', 'rsi', 'momentum']
        datos = [list(map(float, [vela[c] for c in columnas])) for vela in df]
        escalado = scaler_scdp.transform(datos)
        X = np.reshape(escalado, (1, escalado.shape[0], escalado.shape[1]))

        pred = modelo_cierre.predict(X, verbose=0)[0][0]

        if pred > 0.85:
            log(f"[⚠️ SCDP-X] Predicción de pérdida anticipada: {pred:.2f} | Cerrando contrato.")
            registrar_resultado_operacion(-1)  # Simula pérdida, real opcional
            cerrar_contrato_forzado()

    except Exception as e:
        log(f"[❌ ERROR SCDP-X] {e}")

def cerrar_contrato_forzado():
    from core.operaciones import contrato_activo
    # Aquí podría integrarse cierre real vía API si se habilita por Deriv (actualmente no se puede).
    log("[🔒 FORZADO] Contrato marcado como cerrado por SCDP-X.")
    # Eliminar el contrato activo para que se permita una nueva operación
    from core.operaciones import contrato_activo
    contrato_activo = None
