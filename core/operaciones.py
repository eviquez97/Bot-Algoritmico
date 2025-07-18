# core/operaciones.py

import json
import threading
import websocket
import time
from config.vars import TOKEN, APP_ID, SYMBOL
from core.estado import contrato_activo, tiempo_inicio_contrato, datos_operacion
from core.smart_compound import objetivo_diario_alcanzado
from core.vigilancia_contrato import vigilar_contrato_con_websocket
from utils.logs import log

MULTIPLICADORES_PERMITIDOS = [100, 200, 300, 400]
CAPITAL_DISPONIBLE = 400.00
LIMITE_POR_OPERACION = CAPITAL_DISPONIBLE * 0.25

def ejecutar_operacion_put(monto, multiplicador, score_drl, prediccion_futura, ganancia_esperada, duracion_estimada):
    from core.estado import contrato_activo
    if contrato_activo:
        log("[⛔ BLOQUEADO] Ya hay un contrato activo. Esperando cierre para nueva operación.")
        return False

    if objetivo_diario_alcanzado():
        log("[✅ META DIARIA] Ya no se permite operar hoy.")
        return False

    if monto > LIMITE_POR_OPERACION or multiplicador not in MULTIPLICADORES_PERMITIDOS:
        log(f"[❌ INVÁLIDO] Monto o multiplicador no permitidos: Monto={monto} | Mult={multiplicador}")
        return False

    operacion_confirmada = False
    cierre_reportado = False

    def on_message(ws, message):
        nonlocal operacion_confirmada
        try:
            data = json.loads(message)

            if "error" in data:
                log(f"[❌ ERROR WS] {data['error']['message']}")
                ws.close()
                return

            if "buy" in data:
                contrato_id = data["buy"]["contract_id"]
                start_time = data["buy"]["start_time"]
                log(f"🚀 ORDEN EJECUTADA: MULTDOWN | Contrato: {contrato_id}")

                from core.estado import actualizar_estado_contrato
                actualizar_estado_contrato(contrato_id, start_time)

                datos_operacion["monto"] = monto
                datos_operacion["multiplicador"] = multiplicador
                datos_operacion["score"] = score_drl
                datos_operacion["prediccion_futura"] = prediccion_futura
                datos_operacion["ganancia_esperada"] = ganancia_esperada
                datos_operacion["duracion"] = duracion_estimada

                from core.vigilancia_contrato import vigilar_contrato_con_websocket
                hilo_vigilancia = threading.Thread(
                    target=vigilar_contrato_con_websocket,
                    args=(contrato_id,),
                    daemon=True
                )
                hilo_vigilancia.start()

                operacion_confirmada = True
                ws.close()

        except Exception as e:
            log(f"[❌ ERROR WS MSG] {e}")
            ws.close()

    def on_open(ws):
        log("[🔑 WS] Enviando token para autorización...")
        ws.send(json.dumps({"authorize": TOKEN}))

    def on_close(ws, close_status_code=None, close_msg=None):
        nonlocal cierre_reportado
        if not cierre_reportado:
            log(f"[🔌 WS CERRADO] Código: {close_status_code} | Mensaje: {close_msg}")
            cierre_reportado = True

    def on_error(ws, error):
        log(f"[❌ ERROR WS CONEXIÓN] {error}")

    def on_authorize(ws, message):
        log("[✅ AUTHORIZADO] Token aceptado.")
        log("[📥 PROPUESTA] Solicitando propuesta MULTDOWN...")
        ws.send(json.dumps({
            "proposal": 1,
            "amount": monto,
            "basis": "stake",
            "contract_type": "MULTDOWN",
            "currency": "USD",
            "symbol": SYMBOL,
            "multiplier": multiplicador
        }))

    def on_proposal(ws, message):
        data = json.loads(message)
        if "proposal" in data:
            proposal_id = data["proposal"]["id"]
            log("[📤 ORDEN] Enviando orden de compra...")
            ws.send(json.dumps({
                "buy": proposal_id,
                "price": monto
            }))

    def custom_on_message(ws, message):
        data = json.loads(message)
        if "authorize" in data:
            on_authorize(ws, message)
        elif "proposal" in data:
            on_proposal(ws, message)
        else:
            on_message(ws, message)

    ws = websocket.WebSocketApp(
        f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}",
        on_open=on_open,
        on_message=custom_on_message,
        on_error=on_error,
        on_close=on_close
    )

    ws_thread = threading.Thread(target=ws.run_forever)
    ws_thread.start()
    ws_thread.join(timeout=15)

    time.sleep(1)

    if operacion_confirmada:
        return True
    else:
        log("[⚠️ SIN CONFIRMACIÓN] No se recibió confirmación de ejecución de orden.")
        return False


