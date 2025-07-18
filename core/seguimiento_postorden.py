# core/seguimiento_postorden.py

import time
import threading
import requests
from config import log
from core.operaciones import contrato_activo, TOKEN

TIEMPO_MAXIMO_SEGUIMIENTO = 120  # segundos

def obtener_info_contrato(contract_id):
    url = "https://api.deriv.com/api/v1/contract"
    headers = {"Authorization": f"Bearer {TOKEN}"}
    params = {"contract_id": contract_id}

    try:
        response = requests.get(url, headers=headers, params=params)
        return response.json()
    except Exception as e:
        log(f"[❌ SEGUIMIENTO ERROR API] {e}")
        return None

def seguimiento_postorden(contract_id):
    def monitorear():
        tiempo_inicio = time.time()
        while time.time() - tiempo_inicio < TIEMPO_MAXIMO_SEGUIMIENTO:
            info = obtener_info_contrato(contract_id)
            if info and "status" in info and info["status"] == "sold":
                log(f"[✅ CONTRATO CERRADO] {contract_id} se cerró correctamente.")
                return
            time.sleep(5)

        # Si se excede el tiempo
        log(f"[⚠️ CONTRATO ABIERTO >{TIEMPO_MAXIMO_SEGUIMIENTO}s] Forzando cierre...")
        # Aquí podrías añadir lógica de cierre forzado si Deriv lo permite

    thread = threading.Thread(target=monitorear)
    thread.daemon = True
    thread.start()
