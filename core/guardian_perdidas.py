# core/guardian_perdidas.py

import os
import datetime
from utils.logs import log
from core.main_control import detener_bot

RUTA_BALANCE = "data/balance_diario.txt"
LIMITE_PERDIDA_DIARIA = -80.0  # Protección del capital
OBJETIVO_DIARIO = 500.0        # Meta diaria
MAX_RACHA_NEGATIVA = 4

estado_guardian = {
    "ganancia_total": 0.0,
    "racha_negativa": 0
}

def registrar_resultado(ganancia):
    hoy = datetime.date.today().isoformat()

    if os.path.exists(RUTA_BALANCE):
        with open(RUTA_BALANCE, "r") as f:
            lineas = f.readlines()
    else:
        lineas = []

    lineas = [l for l in lineas if not l.startswith(hoy)]

    estado_guardian["ganancia_total"] += ganancia
    if ganancia < 0:
        estado_guardian["racha_negativa"] += 1
    else:
        estado_guardian["racha_negativa"] = 0

    # Guardar resultado actualizado
    lineas.append(f"{hoy},{estado_guardian['ganancia_total']:.2f},{estado_guardian['racha_negativa']}\n")

    with open(RUTA_BALANCE, "w") as f:
        f.writelines(lineas)

    log(f"[📊 GUARDIAN] Ganancia diaria actual: ${estado_guardian['ganancia_total']:.2f} | Racha negativa: {estado_guardian['racha_negativa']}")

    # 🚨 Apagar si se alcanza el objetivo
    if estado_guardian["ganancia_total"] >= OBJETIVO_DIARIO:
        log("[🎯 OBJETIVO DIARIO ALCANZADO] Ganancia alcanzada. Apagando bot para proteger utilidades.")
        detener_bot()

    if estado_guardian["ganancia_total"] <= LIMITE_PERDIDA_DIARIA:
        log("[⛔ APAGADO URGENTE] Pérdida acumulada crítica. Deteniendo bot para proteger capital.")
        detener_bot()

    if estado_guardian["racha_negativa"] >= MAX_RACHA_NEGATIVA:
        log("[⚠️ APAGADO POR RACHA] Racha negativa extrema. Apagando bot por seguridad.")
        detener_bot()

