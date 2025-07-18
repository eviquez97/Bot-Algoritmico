# core/refuerzo_drl.py

from core.ia_modelos import modelo_drl
from core.ia_drl import estado_drl
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
from utils.logs import log

mse = MeanSquaredError()

def reforzar_drl_post_resultado(exito):
    if not estado_drl["evaluaciones"]:
        return

    ultima = estado_drl["evaluaciones"][-1]
    estado = ultima["estado"]
    accion = ultima["accion_index"]
    q_valor = ultima["q_valor"]

    # Recompensa basada en éxito real
    recompensa = 1.0 if exito else -1.0

    try:
        q_actual = modelo_drl.predict(estado, verbose=0)[0]
        q_actual[accion] = recompensa  # Sobreescribimos solo la acción tomada

        modelo_drl.fit(estado, np.array([q_actual]), verbose=0)
        log(f"[🔁 REFORZAMIENTO DRL] Reentrenamiento aplicado post-ronda | {'✅ ÉXITO' if exito else '❌ FALLA'}")

    except Exception as e:
        log(f"[❌ ERROR REFORZAMIENTO DRL] {e}")
