# core/seguridad.py

import time

estado_seguridad = {
    "perdidas_consecutivas": 0,
    "ultima_perdida": 0,
    "operando": True
}

# Se configura a nivel seguro
LIMITE_PERDIDAS_CONSECUTIVAS = 3
LIMITE_PERDIDA_INDIVIDUAL = -150

def registrar_resultado_contrato(ganancia):
    if ganancia < 0:
        estado_seguridad["perdidas_consecutivas"] += 1
        estado_seguridad["ultima_perdida"] = ganancia
    else:
        estado_seguridad["perdidas_consecutivas"] = 0

    if estado_seguridad["perdidas_consecutivas"] >= LIMITE_PERDIDAS_CONSECUTIVAS or ganancia < LIMITE_PERDIDA_INDIVIDUAL:
        estado_seguridad["operando"] = False
        print(f"[🛑 SEGURIDAD] Bot desactivado por pérdidas. Ganancia: {ganancia}, Consecutivas: {estado_seguridad['perdidas_consecutivas']}")
        return False

    return True

def esta_autorizado_operar():
    return estado_seguridad["operando"]
