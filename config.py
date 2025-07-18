import os
from datetime import datetime

# === DATOS DE CUENTA ===
TOKEN = "UurNEj0Vj7c28q1"
APP_ID = "1089"
SYMBOL = "BOOM1000"

# === RUTAS BASE ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELOS = os.path.join(BASE_DIR, "modelos")
RUTA_DATA = os.path.join(BASE_DIR, "data")
RUTA_LOGS = os.path.join(BASE_DIR, "logs")

# === LOG SIMPLE ===
def log(mensaje, nivel="info"):
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{nivel.upper()}] {timestamp} - {mensaje}")

# Combinaciones válidas de (monto, multiplicador)
COMBINACIONES_ACCIONES = [
    (1.00, 100),
    (1.00, 200),
    (1.00, 300),
    (1.00, 400),
    (2.00, 100),
    (2.00, 200),
    (2.00, 300),
    (2.00, 400),
    (3.00, 100),
    (3.00, 200),
]
CONTRACTS_ACTIVOS = {}

# 👇 Control de contrato activo (modo cazador: solo 1 a la vez)
contrato_activo = {
    "id": None,
    "abierto": False,
    "inicio": None
}
tiempo_inicio_contrato = None

# Duración máxima de un contrato (en segundos)
DURACION_MAXIMA_CONTRATO = 180


