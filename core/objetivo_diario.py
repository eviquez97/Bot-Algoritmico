# core/objetivo_diario.py

import os
from datetime import datetime
from utils.logs import log

ARCHIVO_GANANCIAS = "data/ganancias_hoy.txt"
OBJETIVO_DIARIO = 500.0  # Monto objetivo por día en USD

def registrar_ganancia(monto):
    hoy = datetime.now().strftime("%Y-%m-%d")
    try:
        with open(ARCHIVO_GANANCIAS, "a") as f:
            f.write(f"{hoy},{monto}\n")
    except Exception as e:
        log(f"[❌ ERROR REGISTRO GANANCIA] {e}")

def evaluar_ganancias_acumuladas():
    hoy = datetime.now().strftime("%Y-%m-%d")
    if not os.path.exists(ARCHIVO_GANANCIAS):
        return True  # No hay ganancias registradas

    try:
        with open(ARCHIVO_GANANCIAS, "r") as f:
            lineas = f.readlines()

        ganancias_hoy = [float(l.split(",")[1]) for l in lineas if hoy in l]
        total = sum(ganancias_hoy)

        if total >= OBJETIVO_DIARIO:
            log(f"[✅ OBJETIVO ALCANZADO] ${round(total,2)} ganados hoy. Bot detenido por seguridad.")
            return False
        return True
    except Exception as e:
        log(f"[❌ ERROR EVALUACIÓN OBJETIVO] {e}")
        return True
