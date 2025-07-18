import json
import websocket
from config.vars import TOKEN, APP_ID
from utils.logs import log

def verificar_contrato_activo():
    def on_open(ws):
        log("[🔐 VERIFICACIÓN] Autorizando token...")
        ws.send(json.dumps({"authorize": TOKEN}))

    def on_message(ws, message):
        try:
            data = json.loads(message)

            if data.get("msg_type") == "authorize":
                log("[✅ VERIFICACIÓN] Autorización correcta. Solicitando último contrato...")
                ws.send(json.dumps({
                    "statement": 1,
                    "limit": 1
                }))

            elif data.get("msg_type") == "statement":
                statement = data.get("statement")
                if statement and isinstance(statement, list) and len(statement) > 0:
                    ultimo = statement[0]
                    contract_id = ultimo.get("contract_id")
                    log(f"[📊 VERIFICACIÓN] Último contrato encontrado: {contract_id}")
                else:
                    log("[📊 VERIFICACIÓN] No hay contratos recientes en el historial.")
                ws.close()

            else:
                log(f"[📋 VERIFICACIÓN] Mensaje no esperado: {data.get('msg_type')}")
                ws.close()

        except Exception as e:
            log(f"[❌ ERROR WS VERIFICACIÓN] {repr(e)} | Raw: {message}")
            ws.close()

    def on_error(ws, error):
        log(f"[❌ WS ERROR VERIFICACIÓN] {str(error)}")

    def on_close(ws, code, reason):
        log(f"[🔌 WS CERRADO VERIFICACIÓN] Código: {code} | Motivo: {reason}")

    try:
        ws = websocket.WebSocketApp(
            f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}",
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        ws.run_forever()
    except Exception as e:
        log(f"[❌ ERROR GLOBAL VERIFICACIÓN WS] {e}")

def iniciar_verificacion_contratos():
    verificar_contrato_activo()

