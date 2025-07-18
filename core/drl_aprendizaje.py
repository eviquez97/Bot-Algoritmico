import numpy as np
from core.modelos_drl import modelo_drl
from core.modelos_drl import acciones_drl
from utils.logs import log

experiencias_entrenamiento = []

def registrar_experiencia_drl(estado, accion_index, recompensa):
    experiencias_entrenamiento.append((estado, accion_index, recompensa))
    if len(experiencias_entrenamiento) > 200:
        experiencias_entrenamiento.pop(0)

def actualizar_drl_con_recompensa():
    if not experiencias_entrenamiento:
        return

    try:
        estados = []
        targets = []

        for estado, accion_index, recompensa in experiencias_entrenamiento:
            q_vals = modelo_drl.predict(estado, verbose=0)[0]
            q_vals[accion_index] = recompensa  # Aplicar recompensa a esa acción
            estados.append(estado[0][0])  # Quitar batch extra
            targets.append(q_vals)

        X = np.array(estados).reshape(-1, 1, 4)  # (batch, 1, features)
        y = np.array(targets)

        modelo_drl.fit(X, y, epochs=3, verbose=0)
        log(f"[🎓 DRL ENTRENADO] {len(experiencias_entrenamiento)} experiencias aplicadas con recompensas")

        # Limpiar después de entrenar
        experiencias_entrenamiento.clear()

    except Exception as e:
        log(f"[❌ ERROR ENTRENAMIENTO DRL] {e}")
