# core/anti_perdidas.py

import os
from datetime import datetime
from utils.logs import log

LIMITE_PERDIDAS = 4  # Número máximo de pérdidas permitidas antes de bloquear el bot
ARCHIVO_PERDIDAS = "data/perdidas_hoy.txt"

def registrar_resultado_operacion(resultado):
    """
    resultado: debe ser "ganada" o "perdida"
    """
    hoy = datetime.now().strftime("%Y-%m-%d")
    try:
        with open(ARCHIVO_PERDIDAS, "a") as f:
            f.write(f"{hoy},{resultado}\n")
    except Exception as e:
        log(f"[❌ ERROR REGISTRO PÉRDIDA] {e}")

def evaluar_perdidas_acumuladas():
    hoy = datetime.now().strftime("%Y-%m-%d")
    if not os.path.exists(ARCHIVO_PERDIDAS):
        return True  # No hay datos, se permite operar

    try:
        with open(ARCHIVO_PERDIDAS, "r") as f:
            lineas = f.readlines()

        perdidas_hoy = [linea for linea in lineas if hoy in linea and "perdida" in linea]

        if len(perdidas_hoy) >= LIMITE_PERDIDAS:
            log(f"[🛑 BLOQUEO POR PÉRDIDAS] {len(perdidas_hoy)} pérdidas hoy. Se detiene la operativa.")
            return False
        return True
    except Exception as e:
        log(f"[❌ ERROR EVALUACIÓN PÉRDIDAS] {e}")
        return True
