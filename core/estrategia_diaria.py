# core/estrategia_diaria.py

import os
import datetime
from utils.logs import log  # ✅ Corrección aquí
from core.smart_compound import actualizar_scm
from core.guardian_perdidas import registrar_resultado

RUTA_ESTADO = "data/estado_diario.txt"
OBJETIVO_DIARIO = 500  # USD
LIMITE_PERDIDA = -100  # USD

estado = {
    "fecha": "",
    "ganancia": 0,
    "apagado": False
}

def cargar_estado_diario():
    global estado
    hoy = str(datetime.date.today())

    if os.path.exists(RUTA_ESTADO):
        with open(RUTA_ESTADO, "r") as f:
            partes = f.read().strip().split(",")
            if len(partes) == 3 and partes[0] == hoy:
                estado["fecha"] = partes[0]
                estado["ganancia"] = float(partes[1])
                estado["apagado"] = partes[2] == "1"
                return

    estado = {"fecha": hoy, "ganancia": 0, "apagado": False}
    guardar_estado_diario()

def guardar_estado_diario():
    with open(RUTA_ESTADO, "w") as f:
        f.write(f"{estado['fecha']},{estado['ganancia']},{1 if estado['apagado'] else 0}")

def registrar_resultado_operacion(ganancia):
    actualizar_scm(ganancia)  # Sistema de interés compuesto
    registrar_resultado(ganancia)  # Guardian anti-pérdidas

    estado["ganancia"] += ganancia
    log(f"[📈 GANANCIA ACUMULADA] Hoy: ${estado['ganancia']:.2f}")

    if estado["ganancia"] >= OBJETIVO_DIARIO:
        log("🎯 OBJETIVO DIARIO ALCANZADO | BOT DESACTIVADO AUTOMÁTICAMENTE ✅")
        estado["apagado"] = True
    elif estado["ganancia"] <= LIMITE_PERDIDA:
        log("🛑 LÍMITE DE PÉRDIDA SUPERADO | BOT DESACTIVADO AUTOMÁTICAMENTE ❌")
        estado["apagado"] = True

    guardar_estado_diario()

def esta_apagado():
    return estado["apagado"]
