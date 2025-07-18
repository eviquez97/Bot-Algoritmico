# core/riesgo.py

import os

MAX_PERDIDAS_CONSECUTIVAS = 3
MAX_PERDIDA_DIARIA = 200  # USD
VELAS_RECONEXION = 15

estado_riesgo = {
    "perdidas_consecutivas": 0,
    "ganancia_neta": 0,
    "bloqueado": False,
    "velas_sin_operar": 0
}

def registrar_resultado_operacion(ganancia):
    if estado_riesgo["bloqueado"]:
        return

    estado_riesgo["ganancia_neta"] += ganancia

    if ganancia < 0:
        estado_riesgo["perdidas_consecutivas"] += 1
    else:
        estado_riesgo["perdidas_consecutivas"] = 0

    if estado_riesgo["perdidas_consecutivas"] >= MAX_PERDIDAS_CONSECUTIVAS:
        estado_riesgo["bloqueado"] = True
        print("🛑 [RIESGO] Pausa activada: demasiadas pérdidas consecutivas.")

    if estado_riesgo["ganancia_neta"] <= -MAX_PERDIDA_DIARIA:
        estado_riesgo["bloqueado"] = True
        print("🛑 [RIESGO] Pausa activada: pérdida diaria excedida.")

def esta_bloqueado():
    return estado_riesgo["bloqueado"]

def procesar_vela_riesgo():
    if estado_riesgo["bloqueado"]:
        estado_riesgo["velas_sin_operar"] += 1
        if estado_riesgo["velas_sin_operar"] >= VELAS_RECONEXION:
            estado_riesgo["bloqueado"] = False
            estado_riesgo["perdidas_consecutivas"] = 0
            estado_riesgo["ganancia_neta"] = 0
            estado_riesgo["velas_sin_operar"] = 0
            print("✅ [RECUPERACIÓN] Reanudación de operaciones tras pausa.")
