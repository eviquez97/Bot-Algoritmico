# core/registro_operaciones.py

import csv
import os
from datetime import datetime

RUTA_REGISTRO = "data/operaciones_historicas.csv"
ENCABEZADOS = ["timestamp", "open", "high", "low", "close", "spread", "ema", "rsi", "momentum", 
               "score", "ganancia_estim", "porcentaje_bajistas", "pred_futuro", 
               "monto", "multiplicador", "resultado"]

def registrar_operacion(vela, contexto, monto, mult, resultado):
    try:
        datos = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "open": vela.get("open"),
            "high": vela.get("high"),
            "low": vela.get("low"),
            "close": vela.get("close"),
            "spread": vela.get("high") - vela.get("low"),
            "ema": vela.get("ema"),
            "rsi": vela.get("rsi"),
            "momentum": vela.get("momentum"),
            "score": contexto.get("score"),
            "ganancia_estim": contexto.get("ganancia_estim"),
            "porcentaje_bajistas": contexto.get("porcentaje_bajistas"),
            "pred_futuro": contexto.get("pred_futuro"),
            "monto": monto,
            "multiplicador": mult,
            "resultado": resultado  # Puede ser 1 (ganó) o 0 (perdió)
        }

        existe = os.path.exists(RUTA_REGISTRO)
        with open(RUTA_REGISTRO, mode="a", newline='') as archivo:
            escritor = csv.DictWriter(archivo, fieldnames=ENCABEZADOS)
            if not existe:
                escritor.writeheader()
            escritor.writerow(datos)
    except Exception as e:
        print(f"[❌ ERROR REGISTRO OPERACIÓN] {e}")
