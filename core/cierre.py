# core/cierre.py

import json
import threading
import time
import websocket
from config.vars import TOKEN, APP_ID
import core.estado as estado
from utils.logs import log
from core.registro import registrar_ganancia, registrar_resultado_contrato
from utils.deriv_api import obtener_info_contrato_ws

MAX_REINTENTOS_CIERRE = 3
TIEMPO_ENTRE_REINTENTOS = 5  # segundos

def cerrar_contrato_activo():
    if estado.contrato_activo is None and estado.contrato_recuperado is None:
        log("⚠️ [CIERRE] No hay contrato activo para cerrar.")
        return

    contrato_id = estado.contrato_activo or estado.contrato_recuperado

    def vender_contrato(intentos=0):
        def on_open(ws):
            log(f"[🔐 CIERRE] Autenticando intento {intentos + 1} para cerrar contrato...")
            ws.send(json.dumps({"authorize": TOKEN}))

        def on_message(ws, message):
            try:
                data = json.loads(message)

                if data.get("msg_type") == "authorize":
                    log(f"[📤 CIERRE SOLICITADO] Vendiendo contrato ID: {contrato_id}")
                    ws.send(json.dumps({"sell": contrato_id, "price": 0}))

                elif data.get("msg_type") == "sell":
                    log(f"💼 [CIERRE CONFIRMADO] Contrato vendido con éxito: {contrato_id}")
                    profit = data.get("sell", {}).get("profit")

                    if profit is not None:
                        ganancia = round(float(profit), 2)
                    else:
                        log("[⚠️ SIN GANANCIA] No se recibió información de ganancia. Reintentando consulta final...")
                        info = obtener_info_contrato_ws(contrato_id)
                        if info:
                            ganancia = round(float(info.get("profit", 0.0)), 2)
                            log(f"[🔍 CONSULTA FINAL] Ganancia real recuperada: ${ganancia:.2f}")
                        else:
                            ganancia = 0.0

                    if not estado.ganancia_registrada:
                        registrar_ganancia(ganancia)
                        registrar_resultado_contrato(
                            ganancia=ganancia,
                            duracion=0,
                            razon_cierre="✅ Cierre exitoso WS"
                        )
                        estado.ganancia_registrada = True
                    else:
                        log("[ℹ️ OMITIDO] La ganancia ya fue registrada por vigilancia previa.")

                    _limpiar_estado()
                    ws.close()

            except Exception as e:
                log(f"[❌ ERROR WS CIERRE] {e}")
                ws.close()

        def on_error(ws, error):
            log(f"[❌ WS ERROR CIERRE] {error}")
            ws.close()

        def on_close(ws, code, reason):
            log(f"[🔌 WS CIERRE CERRADO] Código: {code} | Motivo: {reason}")
            if estado.contrato_activo or estado.contrato_recuperado:
                if intentos + 1 < MAX_REINTENTOS_CIERRE:
                    log(f"[🔁 REINTENTANDO CIERRE] Intento {intentos + 2} en {TIEMPO_ENTRE_REINTENTOS}s...")
                    time.sleep(TIEMPO_ENTRE_REINTENTOS)
                    vender_contrato(intentos + 1)
                else:
                    log("🛑 [FALLO CIERRE] No se logró vender el contrato tras múltiples intentos.")
                    estado.vigilancia_activada = False

        ws = websocket.WebSocketApp(
            f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}",
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )

        hilo = threading.Thread(target=ws.run_forever)
        hilo.daemon = True
        hilo.start()

    vender_contrato()

def _limpiar_estado():
    estado.contrato_activo = None
    estado.contrato_recuperado = None
    estado.datos_operacion = {}
    estado.vigilancia_activada = False

__all__ = ["cerrar_contrato_activo"]

