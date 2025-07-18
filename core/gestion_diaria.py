# core/gestion_diaria.py

import os
import datetime
from utils.logs import log

RUTA_ESTADO = "data/estado_diario.json"
OBJETIVO_DIA = 500  # Meta diaria en USD
LIMITE_PERDIDA = -80  # Pérdida máxima tolerada por día

estado_diario = {
    "fecha": str(datetime.date.today()),
    "ganancia_total": 0,
    "operaciones": 0,
    "activo": True
}

def cargar_estado_diario():
    if os.path.exists(RUTA_ESTADO):
        import json
        with open(RUTA_ESTADO, "r") as f:
            data = json.load(f)
            if data["fecha"] == str(datetime.date.today()):
                estado_diario.update(data)
            else:
                resetear_estado()
    else:
        resetear_estado()

def resetear_estado():
    estado_diario["fecha"] = str(datetime.date.today())
    estado_diario["ganancia_total"] = 0
    estado_diario["operaciones"] = 0
    estado_diario["activo"] = True
    guardar_estado()

def guardar_estado():
    import json
    with open(RUTA_ESTADO, "w") as f:
        json.dump(estado_diario, f)

def registrar_ganancia(ganancia):
    estado_diario["ganancia_total"] += ganancia
    estado_diario["operaciones"] += 1
    if estado_diario["ganancia_total"] >= OBJETIVO_DIA:
        estado_diario["activo"] = False
        log(f"[🏁 OBJETIVO DIARIO ALCANZADO] Ganancia total: ${estado_diario['ganancia_total']}")
    elif estado_diario["ganancia_total"] <= LIMITE_PERDIDA:
        estado_diario["activo"] = False
        log(f"[🛑 LÍMITE DE PÉRDIDA ALCANZADO] Pérdida total: ${estado_diario['ganancia_total']}")
    guardar_estado()

def se_permite_operar():
    return estado_diario["activo"]
