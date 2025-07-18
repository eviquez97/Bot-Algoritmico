# core/spike_guardian.py

import os
import datetime
from reentrenamiento.entrenador_spike import entrenar_modelo_spike
from reentrenamiento.reentrenador_lstm_spike import reentrenar_modelo_lstm_spike
from utils.logs import log

def verificar_reentrenamiento_spike():
    hoy = datetime.date.today()
    ruta_flag = "reentrenamiento/reentrenado_spike.txt"

    try:
        if os.path.exists(ruta_flag):
            with open(ruta_flag, "r") as f:
                fecha_guardada = f.read().strip()
            if fecha_guardada == str(hoy):
                return  # ✅ Ya se reentrenó hoy
            else:
                log("[♻️ AUTOENTRENAMIENTO] Reentrenando IA de spikes con datos recientes...")
                entrenar_modelo_spike()
                reentrenar_modelo_lstm_spike()
        else:
            log("[📅 PRIMER ENTRENAMIENTO HOY] No hay flag, reentrenando spike IA...")
            entrenar_modelo_spike()
            reentrenar_modelo_lstm_spike()

        with open(ruta_flag, "w") as f:
            f.write(str(hoy))
        log("[✅ ENTRENAMIENTO SPIKE COMPLETO] Flag actualizado.")

    except Exception as e:
        log(f"[❌ ERROR REENTRENAMIENTO SPIKE] {e}")

