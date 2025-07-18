# core/websocket_orden.py

import json
import time
import websocket
import threading
from datetime import datetime
from config import TOKEN, APP_ID, SYMBOL, CONTRACTS_ACTIVOS
from utils.logs import log
from core.estado import (
    contrato_activo,
    tiempo_inicio_contrato,
    datos_operacion,
    duracion_contrato_segundos,
    vigilancia_activada
)
from core.watchdog import iniciar_watchdog

def enviar_orden_ws(monto, multiplicador, contexto):
    try:
        def on_open(ws):
            try:
                proposal_data = {
                    "proposal": 1,
                    "amount": monto,
                    "basis": "stake",
                    "contract_type": "MULTIDOWN",
                    "currency": "USD",
                    "symbol": SYMBOL,
                    "multiplier": multiplicador
                }
                ws.send(json.dumps({"authorize": TOKEN}))
                time.sleep(1)
                ws.send(json.dumps(proposal_data))
                log(f"[📡 WS ABIERTA] Enviando orden: ${monto} x{multiplicador}")
            except Exception as e:
                log(f"[❌ ERROR OPEN WS] {e}", "error")
                ws.close()

        def on_message(ws, message):
            try:
                data = json.loads(message)

                if "proposal" in data:
                    proposal_id = data["proposal"]["id"]
                    precio = data["proposal"]["ask_price"]
                    log(f"[📄 PROPOSAL] ID: {proposal_id} | Precio: ${precio}")
                    ws.send(json.dumps({"buy": proposal_id, "price": precio}))

                elif "buy" in data:
                    contrato_id = data["buy"].get("contract_id", "desconocido")
                    log(f"[🚀 ORDEN EJECUTADA] Contrato ID: {contrato_id}")

                    # ================================
                    # 🧠 REGISTRO EN ESTADO GLOBAL
                    # ================================
                    from core import estado as est
                    est.contrato_activo = contrato_id
                    est.tiempo_inicio_contrato = int(datetime.utcnow().timestamp())
                    est.vigilancia_activada = False
                    est.duracion_contrato_segundos = contexto.get("tiempo_estimado", 90)
                    est.datos_operacion.update({
                        "monto": monto,
                        "multiplicador": multiplicador,
                        "score": contexto.get("score", 0),
                        "prediccion_futura": contexto.get("pred_futuro", 0),
                        "ganancia_esperada": contexto.get("ganancia_estim", 0),
                    })

                    # Activar vigilancia tras asignar estado
                    iniciar_watchdog()

                    CONTRACTS_ACTIVOS[contrato_id] = {
                        "tipo": "PUT",
                        "monto": monto,
                        "multiplicador": multiplicador,
                        "precio": contexto.get("precio_entrada", 0),
                        "score": contexto.get("score", 0),
                        "ganancia_max": 0,
                        "target_profit": 0.30,
                        "tiempo_estimado": est.duracion_contrato_segundos,
                        "tiempo_entrada": datetime.utcnow(),
                        "estado_vector": contexto.get("estado_vector"),
                        "accion_idx": contexto.get("accion_idx"),
                        "spike_detectado_post": False
                    }
                    ws.close()

                elif "error" in data:
                    log(f"[❌ DERIV ERROR] {data['error']['message']}", "error")
                    ws.close()

            except Exception as e:
                log(f"[❌ ERROR MENSAJE WS] {e}", "error")
                ws.close()

        def on_error(ws, error):
            log(f"[❌ ERROR WS] {error}", "error")

        def on_close(ws, close_status_code, close_msg):
            log(f"[🔌 WS CERRADO] Código: {close_status_code} | Mensaje: {close_msg}")

        def mantener_conexion(ws):
            while ws.keep_running:
                try:
                    ws.send(json.dumps({"ping": 1}))
                    time.sleep(15)
                except:
                    break

        ws = websocket.WebSocketApp(
            f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}",
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )

        threading.Thread(target=mantener_conexion, args=(ws,), daemon=True).start()
        ws.run_forever()

    except Exception as e:
        log(f"[❌ ERROR ENVÍO ORDEN] {e}", "error")

def cerrar_contrato(contrato_id):
    log(f"[🔒 CIERRE FORZADO] Se intentó cerrar contrato ID: {contrato_id} desde Watchdog")
    # Aquí podrías añadir lógica real para cierre si tu broker lo permite.


