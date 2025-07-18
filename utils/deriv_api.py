# utils/deriv_api.py

import websocket
import json
import ssl
import time
from threading import Thread

# 🔐 Configuración Deriv API
TOKEN = "UurNEj0Vj7c28q1"
APP_ID = "1089"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Origin": "https://deriv.com"
}

# ========================================
# ✅ Función común para construir WebSocket
# ========================================
def crear_websocket_app_deriv(on_open, on_message, on_error, on_close):
    return websocket.WebSocketApp(
        f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}",
        header=[f"{k}: {v}" for k, v in HEADERS.items()],
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

# 📡 Obtener info del contrato (bloqueante)
def obtener_info_contrato(contract_id, intentos=5, timeout=10):
    for intento in range(intentos):
        try:
            resultado = {}
            finalizado = False

            def on_open(ws):
                ws.send(json.dumps({"authorize": TOKEN}))

            def on_message(ws, message):
                nonlocal resultado, finalizado
                data = json.loads(message)

                if data.get("msg_type") == "authorize":
                    ws.send(json.dumps({
                        "proposal_open_contract": 1,
                        "contract_id": contract_id
                    }))
                elif data.get("msg_type") == "proposal_open_contract":
                    resultado = data["proposal_open_contract"]
                    finalizado = True
                    ws.close()
                elif "error" in data:
                    print(f"[❌ ERROR API DERIV] {data['error'].get('message')}")
                    finalizado = True
                    ws.close()

            def on_error(ws, error):
                print(f"[❌ WS ERROR] {error}")
                ws.close()

            def on_close(ws, code, reason):
                print(f"[🔌 WS CERRADO] Código: {code} | Motivo: {reason}")

            ws = crear_websocket_app_deriv(on_open, on_message, on_error, on_close)

            hilo = Thread(target=ws.run_forever, kwargs={"sslopt": {"cert_reqs": ssl.CERT_NONE}})
            hilo.daemon = True
            hilo.start()

            inicio = time.time()
            while not finalizado and (time.time() - inicio < timeout):
                time.sleep(0.1)

            if resultado:
                return resultado

            print(f"[⚠️ INTENTO {intento + 1}] Sin éxito. Reintentando...")
            time.sleep(2)

        except Exception as e:
            print(f"[❌ EXCEPCIÓN obtener_info_contrato] {e}")
            time.sleep(2)

    print("[🛑 ERROR FINAL] No se pudo obtener info del contrato tras varios intentos.")
    return None

# 📡 Versión para vigilancia (intercambiable con `vigilar_contrato_con_websocket`)
def obtener_info_contrato_ws(contract_id, intentos=5, timeout=10):
    for intento in range(intentos):
        try:
            resultado = {}
            finalizado = False

            def on_open(ws):
                ws.send(json.dumps({"authorize": TOKEN}))

            def on_message(ws, message):
                nonlocal resultado, finalizado
                data = json.loads(message)

                if data.get("msg_type") == "authorize":
                    ws.send(json.dumps({
                        "proposal_open_contract": 1,
                        "contract_id": contract_id
                    }))
                elif data.get("msg_type") == "proposal_open_contract":
                    resultado = data["proposal_open_contract"]
                    finalizado = True
                    ws.close()
                elif "error" in data:
                    print(f"[❌ ERROR WS] {data['error'].get('message')}")
                    finalizado = True
                    ws.close()

            def on_error(ws, error):
                print(f"[❌ WS ERROR] {error}")
                ws.close()

            def on_close(ws, code, reason):
                print(f"[🔌 WS CERRADO] Código: {code} | Motivo: {reason}")

            ws = crear_websocket_app_deriv(on_open, on_message, on_error, on_close)

            hilo = Thread(target=ws.run_forever, kwargs={"sslopt": {"cert_reqs": ssl.CERT_NONE}})
            hilo.daemon = True
            hilo.start()

            inicio = time.time()
            while not finalizado and (time.time() - inicio < timeout):
                time.sleep(0.1)

            if resultado:
                return resultado

            print(f"[⚠️ WS INTENTO {intento + 1}] Sin éxito.")
            time.sleep(2)

        except Exception as e:
            print(f"[❌ EXCEPCIÓN obtener_info_contrato_ws] {e}")
            time.sleep(2)

    print("[🛑 WS ERROR FINAL] No se obtuvo info del contrato.")
    return None

# ✅ Construir mensaje de suscripción a ticks (¡sin req_id!)
def construir_subscripcion_ticks(symbol: str = "boom1000"):
    return {
        "ticks": symbol,
        "subscribe": 1
    }

# ❌ Cancelar suscripción a canal (si lo necesitás)
def cancelar_subscripcion_ticks(req_id: str):
    return {
        "forget": req_id
    }

# 🔐 Autorización
def construir_autorizacion():
    return {
        "authorize": TOKEN
    }

