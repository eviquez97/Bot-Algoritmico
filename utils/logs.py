# utils/logs.py

import datetime

# Modos posibles: "verbose", "operativo", "silencioso"
MODO_LOG = "operativo"

def log(mensaje: str, nivel: str = "INFO"):
    filtros_operativo = [
        "DRL", "AUTOENTRENAMIENTO", "AUTO FUTURO", "VERIFICACIÓN", "LSTM"
        # ⚠️ "SPIKE IA" ha sido excluido del filtro para permitir visibilidad completa
    ]

    if MODO_LOG == "silencioso" and nivel != "ERROR":
        return

    if MODO_LOG == "operativo":
        if any(etiqueta in mensaje for etiqueta in filtros_operativo):
            if "SPIKE IA" not in mensaje:  # Permitir logs de SPIKE IA incluso si tienen LSTM
                return

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{nivel}] {timestamp} - {mensaje}")
