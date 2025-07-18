import json
import websocket
import threading
import time
from utils.logs import log
from core.estrategia_diaria import registrar_resultado_operacion
from core.scpx_predictivo import evaluar_cierre_predictivo_total

TOKEN = "UurNEj0Vj7c28q1"
APP_ID = "1089"

def monitorear_cierre_contrato():
    from core.operaciones import contrato_activo

    if not contrato_activo:
        log("⚠️ SEGUIMIENTO] No hay contrato activo para monitorear.")
        return

    def seguimiento():
        ws = websocket.WebSocket()
        try:
            ws.connect(f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}")
            ws.send(json.dumps({"authorize": TOKEN}))
            autorizado = False

            while True:
                try:
                    msg = json.loads(ws.recv())
                except Exception as e:
                    log(f"[❌ ERROR WEBSOCKET RECV] {e}")
                    break

                if not autorizado and "authorize" in msg:
                    ws.send(json.dumps({
                        "proposal_open_contract": 1,
                        "contract_id": contrato_activo
                    }))
                    autorizado = True

                elif "proposal_open_contract" in msg:
                    contrato = msg["proposal_open_contract"]

                    if contrato.get("is_sold"):
                        ganancia = float(contrato.get("profit", 0))
                        log(f"[✅ CONTRATO CERRADO] Ganancia: ${ganancia}")
                        registrar_resultado_operacion(ganancia)
                        break

                    # 🧠 Evaluación de cierre predictivo por IA
                    try:
                        if evaluar_cierre_predictivo_total(contrato):
                            ws.send(json.dumps({"sell": contrato_activo, "price": 0}))
                            log("[⚠️ CIERRE ANTICIPADO] Activado por IA predictiva SCDP-X.")
                            break
                    except Exception as e:
                        log(f"[❌ ERROR EVALUACIÓN CIERRE] {e}")

                time.sleep(3)

        except Exception as e:
            log(f"[❌ ERROR MONITOREO CONTRATO] {e}")
        finally:
            try:
                ws.close()
            except:
                pass
            log("[🔚 MONITOREO FINALIZADO] WebSocket cerrado.")

    hilo = threading.Thread(target=seguimiento)
    hilo.daemon = True
    hilo.start()


