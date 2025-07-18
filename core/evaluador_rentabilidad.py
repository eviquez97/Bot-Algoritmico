# core/evaluador_rentabilidad.py

import csv
import os
from datetime import datetime
from config import log

RUTA_RESULTADOS = "data/resultados_ronda.csv"

def registrar_resultado(ganancia, fue_exito):
    existe = os.path.exists(RUTA_RESULTADOS)
    with open(RUTA_RESULTADOS, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not existe:
            writer.writerow(["timestamp", "ganancia", "exito"])  # encabezado
        writer.writerow([datetime.now().isoformat(), ganancia, int(fue_exito)])

def evaluar_rentabilidad_total():
    if not os.path.exists(RUTA_RESULTADOS):
        return {"ganancia_total": 0, "aciertos": 0, "errores": 0, "efectividad": 0.0}

    total, aciertos, errores = 0, 0, 0
    with open(RUTA_RESULTADOS, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += float(row["ganancia"])
            if row["exito"] == "1":
                aciertos += 1
            else:
                errores += 1

    total_intentos = aciertos + errores
    efectividad = (aciertos / total_intentos * 100) if total_intentos > 0 else 0.0

    return {
        "ganancia_total": round(total, 2),
        "aciertos": aciertos,
        "errores": errores,
        "efectividad": round(efectividad, 2)
    }

def imprimir_estadisticas():
    stats = evaluar_rentabilidad_total()
    log(f"[📊 ESTADÍSTICAS] Ganancia Total: ${stats['ganancia_total']} | Aciertos: {stats['aciertos']} | Errores: {stats['errores']} | Efectividad: {stats['efectividad']}%")
