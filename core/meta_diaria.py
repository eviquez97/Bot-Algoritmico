# core/meta_diaria.py

import os
from datetime import date
from utils.logs import log

ruta_archivo = "data/rendimiento_diario.txt"
META_DIARIA = 500
LIMITE_PERDIDA = -80

ganancia_total = 0  # Acumulador en tiempo real

def registrar_ganancia(monto):
    global ganancia_total
    ganancia_total += monto
    log(f"[📈 META DIARIA] Ganancia acumulada: ${ganancia_total:.2f}")

def objetivo_diario_alcanzado():
    return ganancia_total >= META_DIARIA

def registrar_resultado_dia(ganancia_neta):
    hoy = str(date.today())
    linea = f"{hoy},{ganancia_neta}\n"

    if not os.path.exists("data"):
        os.makedirs("data")

    with open(ruta_archivo, "a") as f:
        f.write(linea)

def evaluar_meta_o_perdida_dia():
    if not os.path.exists(ruta_archivo):
        return False  # Nada que evaluar aún

    hoy = str(date.today())
    ganancias = 0

    with open(ruta_archivo, "r") as f:
        for linea in f:
            fecha, ganancia = linea.strip().split(",")
            if fecha == hoy:
                ganancias += float(ganancia)

    log(f"[📊 RENDIMIENTO HOY] {ganancias} USD")

    if ganancias >= META_DIARIA:
        log("[✅ META DIARIA ALCANZADA] Bot se detiene para asegurar ganancias.")
        return True

    if ganancias <= LIMITE_PERDIDA:
        log("[❌ PÉRDIDA CRÍTICA DETECTADA] Bot se apaga para proteger capital.")
        return True

    return False

