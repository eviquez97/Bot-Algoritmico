# recolector_spike_monstruo.py

import os
import json
import threading
import websocket
import pandas as pd
import ssl
import time

TOKEN = "UurNEj0Vj7c28q1"
SYMBOL = "BOOM1000"
ARCHIVO_SALIDA = "data/dataset_spike_monstruo.csv"

ticks_buffer = []
vela_actual = {
    "open": None,
    "high": float("-inf"),
    "low": float("inf"),
    "close": None,
    "epoch_inicio": None
}

if not os.path.exists("data"):
    os.makedirs("data")

if os.path.exists(ARCHIVO_SALIDA):
    df_dataset = pd.read_csv(ARCHIVO_SALIDA)
else:
    df_dataset = pd.DataFrame()

def calcular_indicadores(df):
    df["spread"] = df["high"] - df["low"]
    df["ema"] = df["close"].ewm(span=10, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + df["close"].diff().apply(lambda x: max(x, 0)).rolling(window=14).mean() /
                                df["close"].diff().apply(lambda x: abs(min(x, 0))).rolling(window=14).mean()))
    df["momentum"] = df["close"] - df["close"].shift(4)
    return df

def detectar_spike(vela):
    rango = vela["high"] - vela["low"]
    return 1 if rango > 1.5 else 0

def cerrar_y_guardar_vela():
    global vela_actual, df_dataset

    vela_actual["close"] = ticks_buffer[-1]["quote"]
    vela_actual["high"] = max(vela_actual["high"], vela_actual["close"])
    vela_actual["low"] = min(vela_actual["low"], vela_actual["close"])
    vela_actual["epoch"] = vela_actual["epoch_inicio"] + 60

    df_vela = pd.DataFrame([vela_actual])
    df_vela = calcular_indicadores(df_vela)

    if df_vela.isnull().values.any():
        print("[⚠️ DESCARTADA] Vela con indicadores incompletos.")
    else:
        spike = detectar_spike(vela_actual)
        df_vela["spike"] = spike
        df_dataset = pd.concat([df_dataset, df_vela], ignore_index=True)
        df_dataset.to_csv(ARCHIVO_SALIDA, index=False)
        print(f"[💾 GUARDADA] Vela añadida al dataset | Spike: {spike}")

    vela_actual.update({
        "open": None,
        "high": float("-inf"),
        "low": float("inf"),
        "close": None,
        "epoch_inicio": None
    })

def on_message(ws, message):
    global ticks_buffer, vela_actual
    data = json.loads(message)
    if "tick" in data:
        tick = data["tick"]
        ticks_buffer.append(tick)

        ts = tick["epoch"]
        quote = tick["quote"]

        if vela_actual["open"] is None:
            vela_actual["open"] = quote
            vela_actual["epoch_inicio"] = ts

        vela_actual["high"] = max(vela_actual["high"], quote)
        vela_actual["low"] = min(vela_actual["low"], quote)

        if ts >= vela_actual["epoch_inicio"] + 60:
            cerrar_y_guardar_vela()

def on_open(ws):
    ws.send(json.dumps({"authorize": TOKEN}))

def on_message_wrapper(ws, message):
    data = json.loads(message)
    if "authorize" in data:
        ws.send(json.dumps({"ticks": SYMBOL, "subscribe": 1}))
        print("[📡 SUBSCRITO] Ticks en tiempo real.")
    else:
        on_message(ws, message)

def on_error(ws, error):
    print(f"[❌ ERROR] {error}")

def on_close(ws, close_status_code, close_msg):
    print("[🔌 DESCONECTADO] WebSocket cerrado.")

def iniciar_recoleccion():
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp(
        "wss://ws.deriv.com/websockets/v3",
        on_open=on_open,
        on_message=on_message_wrapper,
        on_error=on_error,
        on_close=on_close,
        header={"User-Agent": "Mozilla/5.0"}
    )
    ws.run_forever(
        ping_interval=30,
        ping_timeout=10,
        sslopt={"cert_reqs": ssl.CERT_NONE}
    )

print("[🚀 INICIANDO] Recolector de velas y detección de spikes...")
iniciar_recoleccion()



