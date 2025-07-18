# core/ws_ticks.py
import websocket
import json
import threading
import time
from datetime import datetime
import pandas as pd

from utils.logs import log
from core.procesamiento import procesar_vela
from core.autocuracion import marcar_tick_recibido

# Configuración
TOKEN = "UurNEj0Vj7c28q1"
SYMBOL = "BOOM1000"
APP_ID = "1089"

ticks_acumulados = []
minuto_actual = None
autorizado = False

def on_message(ws, message):
    global autorizado, ticks_acumulados, minuto_actual
    try:
        data = json.loads(message)

        if "authorize" in data:
            autorizado = True
            log("[🔑 AUTORIZADO] Token aceptado, suscribiendo a ticks...")
            ws.send(json.dumps({
                "ticks": SYMBOL,
                "subscribe": 1
            }))
            return

        if isinstance(data, dict) and "tick" in data:
            tick = data["tick"]
            if tick and "epoch" in tick and "quote" in tick:
                marcar_tick_recibido()
                epoch = tick["epoch"]
                quote = tick["quote"]
                tick_time = datetime.utcfromtimestamp(epoch).replace(second=0, microsecond=0)

                if minuto_actual is None:
                    minuto_actual = tick_time

                if tick_time > minuto_actual:
                    if ticks_acumulados:
                        precios = [t["quote"] for t in ticks_acumulados]
                        vela = {
                            "epoch": int(minuto_actual.timestamp()),
                            "open": precios[0],
                            "high": max(precios),
                            "low": min(precios),
                            "close": precios[-1]
                        }
                        procesar_vela(vela)  # ✅ Solo esto
                    ticks_acumulados = []
                    minuto_actual = tick_time

                ticks_acumulados.append({"epoch": epoch, "quote": quote})
            else:
                log("[⚠️ TICK DESCARTADO] Tick inválido o incompleto")

    except Exception as e:
        log(f"[❌ ERROR PROCESAMIENTO MENSAJE] {e}")

def on_error(ws, error):
    log(f"[❌ WS ERROR] {error}")

def on_close(ws, close_status_code, close_msg):
    log(f"[❌ WS CERRADO] Código: {close_status_code} | Mensaje: {close_msg}")
    log("[🔁 REINTENTANDO] Reconectando en 5 segundos...")
    time.sleep(5)
    iniciar_ws_ticks()

def on_open(ws):
    try:
        log("[🔐 AUTORIZANDO] Enviando token de acceso...")
        ws.send(json.dumps({
            "authorize": TOKEN
        }))
    except Exception as e:
        log(f"[❌ ERROR AL ENVIAR MENSAJES WS] {e}")

def iniciar_ws_ticks():
    try:
        ws = websocket.WebSocketApp(
            f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}",
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        thread = threading.Thread(target=ws.run_forever)
        thread.daemon = True
        thread.start()
        log("[📡 WS ABIERTO] Conexión establecida.")
    except Exception as e:
        log(f"[❌ ERROR WS INIT] {e}")



