# watchdog.py

import threading
import time
import websocket
import json
import pandas as pd
from utils.logs import log
from config.vars import APP_ID, TOKEN
from core.estado import contrato_activo, tiempo_inicio_contrato, datos_operacion
from core.cierre import cerrar_contrato_activo
from core.ia_spike import evaluar_spike_ia

INTERVALO_VERIFICACION = 5     # segundos entre chequeos
MARGEN_TIEMPO_EXTRA = 30       # margen adicional de gracia en segundos
CSV_HISTORICO = "data/contexto_historico.csv"

def consultar_estado_contrato(contract_id):
    try:
        respuesta = {}

        def on_open(ws):
            ws.send(json.dumps({"authorize": TOKEN}))

        def on_message(ws, message):
            data = json.loads(message)
            if "authorize" in data:
                ws.send(json.dumps({
                    "proposal_open_contract": 1,
                    "contract_id": contract_id
                }))
            elif "proposal_open_contract" in data:
                respuesta["is_expired"] = data["proposal_open_contract"].get("is_expired", True)
                respuesta["is_valid_to_sell"] = data["proposal_open_contract"].get("is_valid_to_sell", False)
                ws.close()

        def on_error(ws, error):
            log(f"[❌ ERROR WS ESTADO] {error}")
            ws.close()

        def on_close(ws, code, reason):
            pass

        ws = websocket.WebSocketApp(
            f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}",
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )

        hilo = threading.Thread(target=ws.run_forever)
        hilo.start()
        hilo.join(timeout=10)
        time.sleep(1)
        return respuesta.get("is_expired", True)

    except Exception as e:
        log(f"[❌ ERROR CONSULTA ESTADO] {e}")
        return True

def watchdog_loop():
    while True:
        try:
            if contrato_activo and tiempo_inicio_contrato:
                # 🧠 Evaluación de Spike IA antes de timeout
                try:
                    df = pd.read_csv(CSV_HISTORICO, on_bad_lines="skip").dropna().tail(60)
                    resultado_spike = evaluar_spike_ia(df)
                    if resultado_spike.get("bloqueado", False):
                        log("💣 [WATCHDOG] Spike IA anticipado. Cierre forzado inmediato.")
                        cerrar_contrato_activo()
                        time.sleep(INTERVALO_VERIFICACION)
                        continue
                except Exception as e:
                    log(f"[⚠️ ERROR EVALUANDO SPIKE EN WATCHDOG] {e}")

                # ⏳ Evaluación por duración
                tiempo_actual = time.time()
                duracion_transcurrida = tiempo_actual - tiempo_inicio_contrato
                duracion_esperada = datos_operacion.get("duracion")

                if duracion_esperada is None:
                    log("[⚠️ WATCHDOG] Duración estimada no disponible.")
                    time.sleep(INTERVALO_VERIFICACION)
                    continue

                if duracion_transcurrida > duracion_esperada + MARGEN_TIEMPO_EXTRA:
                    log("⏱️ [⌛ SUPERVISIÓN] Verificando si el contrato sigue activo...")
                    contrato_expirado = consultar_estado_contrato(contrato_activo)

                    if not contrato_expirado:
                        log("🛑 [CIERRE FORZADO] El contrato sigue abierto tras el tiempo límite.")
                        cerrar_contrato_activo()
                    else:
                        log("✅ [VIGILANCIA] Contrato ya finalizado. No se requiere cierre.")

        except Exception as e:
            log(f"[❌ ERROR WATCHDOG LOOP] {e}")

        time.sleep(INTERVALO_VERIFICACION)

def iniciar_watchdog():
    thread = threading.Thread(target=watchdog_loop, daemon=True)
    thread.start()
    log("[🧩 WATCHDOG] Vigilancia inteligente iniciada correctamente.")

