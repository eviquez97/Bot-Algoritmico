# recolector_dataset_operativo.py

import websocket
import json
import time
import threading
import ssl
import core.estado as estado
from core.estado import actualizar_tick
from datetime import datetime
from core.procesamiento import procesar_vela
from core.ia_spike import cargar_modelos_spike
from core.ia_drl import cargar_modelo_drl_ganancia
from core.ia_cierre import cargar_modelo_cierre
from utils.logs import log
from config.vars import TOKEN, APP_ID, SYMBOL

# =========================
# 🔁 Carga inicial de modelos
# =========================
cargar_modelos_spike()
cargar_modelo_drl_ganancia()
cargar_modelo_cierre()

# =========================
# 🌐 WebSocket Deriv
# =========================
URL = f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}"
RECONNECT_INTERVAL = 5

def enviar_autenticacion(ws):
    ws.send(json.dumps({"authorize": TOKEN}))
    log("[🔑 WS] Enviando token para autorización...")

def suscribir_ticks(ws):
    ws.send(json.dumps({
        "ticks": SYMBOL,
        "subscribe": 1
    }))
    log(f"[📡 WS] Suscrito a ticks en vivo de {SYMBOL}.")

def on_open(ws):
    log("[✅ WS] Conexión abierta.")
    enviar_autenticacion(ws)

def on_message(ws, message):
    try:
        data = json.loads(message)

        if 'error' in data:
            log(f"[❌ ERROR WS] {data['error'].get('message')}")
            return

        if data.get("msg_type") == "authorize":
            log("[✅ AUTHORIZADO] Token aceptado.")
            suscribir_ticks(ws)

        elif data.get("msg_type") == "tick":
            tick = data["tick"]
            actualizar_tick(tick)
    except Exception as e:
        log(f"[❌ ERROR on_message] {e}")

def on_error(ws, error):
    log(f"[❌ ERROR WS] {error}")

def on_close(ws, close_status_code, close_msg):
    log(f"[🔌 WS CERRADO] Código: {close_status_code} | Mensaje: {close_msg}")

def mantener_conexion():
    while True:
        try:
            ws = websocket.WebSocketApp(
                URL,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        except Exception as e:
            log(f"[❌ ERROR CONEXIÓN] {e}")
        log(f"[🔁 REINTENTO] Reconectando en {RECONNECT_INTERVAL}s...")
        time.sleep(RECONNECT_INTERVAL)

def construir_velas():
    vela_actual = None
    volumen = 0

    while True:
        tick = estado.ultimo_tick
        if not tick:
            time.sleep(0.5)
            continue

        epoch = tick["epoch"]
        precio = tick["quote"]
        dt = datetime.utcfromtimestamp(epoch)
        segundo = dt.second

        if vela_actual is None:
            vela_actual = {
                "open": precio,
                "high": precio,
                "low": precio,
                "close": precio,
                "volumen_tick": 1,
                "timestamp_inicio": epoch
            }
            volumen = 1
            continue

        vela_actual["high"] = max(vela_actual["high"], precio)
        vela_actual["low"] = min(vela_actual["low"], precio)
        vela_actual["close"] = precio
        volumen += 1
        vela_actual["volumen_tick"] = volumen

        if segundo == 59:
            procesar_vela(vela_actual.copy())
            vela_actual = None
            volumen = 0
            time.sleep(1.2)

        time.sleep(0.2)

if __name__ == "__main__":
    log("[🚀 INICIANDO RECOLECTOR]")
    threading.Thread(target=mantener_conexion, daemon=True).start()
    construir_velas()

