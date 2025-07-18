# core/scpx_predictivo.py

import numpy as np
from utils.logs import log
from modelos.modelos_cierre import modelo_cierre, scaler_cierre

def evaluar_cierre_predictivo_total(contrato):
    """
    Evalúa si un contrato debe cerrarse anticipadamente basado en predicción IA.
    Retorna True si se debe cerrar ya mismo.
    """

    try:
        # ⚙️ Validar si el contrato está en pérdida
        profit = float(contrato.get("profit", 0))
        if profit >= 0:
            return False  # No se cierra si no está en pérdida

        # 🧠 Extraer features para predicción
        datos = {
            "buy_price": float(contrato.get("buy_price", 0)),
            "bid_price": float(contrato.get("bid_price", 0)),
            "profit": profit,
            "entry_tick": float(contrato.get("entry_tick", 0)),
            "exit_tick": float(contrato.get("exit_tick", 0)),
            "high_barrier": float(contrato.get("high_barrier", 0)),
            "low_barrier": float(contrato.get("low_barrier", 0)),
            "current_spot": float(contrato.get("current_spot", 0)),
            "payout": float(contrato.get("payout", 0))
        }

        X = np.array([[datos[k] for k in datos]])
        X_scaled = scaler_cierre.transform(X)
        pred = modelo_cierre.predict(X_scaled, verbose=0)[0][0]

        log(f"[🔬 SCPX] Predicción de cierre: {round(pred, 4)}")

        if pred >= 0.65:  # Umbral de cierre anticipado
            return True
        return False

    except Exception as e:
        log(f"[❌ ERROR SCPX] Evaluación de cierre fallida: {e}")
        return False
