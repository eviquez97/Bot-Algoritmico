import json
import time
import websocket
from datetime import datetime
from config import APP_ID, TOKEN, CONTRACTS_ACTIVOS
from utils.log import log
from core.spike_guardian_predictivo import evaluar_spike_anticipado
from core.registro import registrar_operacion_ia, guardar_operacion, manejar_resultado_contrato

# === GESTOR DE CONTRATO ACTIVO ===
def gestionar_contrato(contrato_id):
    try:
        if contrato_id not in CONTRACTS_ACTIVOS:
            log(f"[⚠️ CONTRATO DESCONOCIDO] {contrato_id}")
            return

        ws = websocket.WebSocket()
        ws.connect(f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}")
        ws.send(json.dumps({"authorize": TOKEN}))
        time.sleep(1)
        ws.send(json.dumps({"proposal_open_contract": 1, "contract_id": contrato_id}))

        info = CONTRACTS_ACTIVOS[contrato_id]
        profit_max = 0
        objetivo = info.get("target_profit", 0.30)
        tiempo_estimado = info.get("tiempo_estimado", 90)
        inicio = info["tiempo_entrada"]
        spike_defendido = False

        while True:
            ws.send(json.dumps({"ping": 1}))
            raw = ws.recv()
            data = json.loads(raw)

            if "proposal_open_contract" in data:
                poc = data["proposal_open_contract"]
                if poc.get("is_sold") or poc.get("is_expired"):
                    log(f"[📤 CONTRATO FINALIZADO] {contrato_id}")
                    break

                profit = float(poc.get("profit", 0.0))
                bid_price = poc.get("bid_price", 0.0)
                duracion = (datetime.utcnow() - inicio).total_seconds()

                # 📈 Cierre por objetivo alcanzado
                if profit >= objetivo:
                    ws.send(json.dumps({"sell": contrato_id, "price": bid_price}))
                    log(f"[✅ OBJETIVO ALCANZADO] +${profit:.2f}")
                    registrar_operacion_ia(info["score"], objetivo, profit, int(duracion), 1)
                    guardar_operacion(contrato_id, info["score"], info["multiplicador"], info["monto"], objetivo, tiempo_estimado, int(duracion), profit, False, "ganada")
                    manejar_resultado_contrato({"contract_id": contrato_id, "profit": profit})
                    break

                # 🛡️ Cierre defensivo por spike
                if detectar_spike_inminente() and not spike_defendido and profit >= 0.01:
                    ws.send(json.dumps({"sell": contrato_id, "price": bid_price}))
                    log(f"[⚠️ SPIKE DETECTADO] +${profit:.2f} → Cierre defensivo")
                    registrar_operacion_ia(info["score"], objetivo, profit, int(duracion), 1)
                    guardar_operacion(contrato_id, info["score"], info["multiplicador"], info["monto"], objetivo, tiempo_estimado, int(duracion), profit, True, "ganada")
                    manejar_resultado_contrato({"contract_id": contrato_id, "profit": profit})
                    spike_defendido = True
                    break

                # 🕒 Cierre por estancamiento
                if duracion > tiempo_estimado + 30 and profit >= 0.01:
                    ws.send(json.dumps({"sell": contrato_id, "price": bid_price}))
                    resultado = "ganada" if profit >= objetivo else "neutra"
                    log(f"[⏱️ CIERRE TARDÍO] ${profit:.2f} | Resultado: {resultado.upper()}")
                    registrar_operacion_ia(info["score"], objetivo, profit, int(duracion), int(resultado == "ganada"))
                    guardar_operacion(contrato_id, info["score"], info["multiplicador"], info["monto"], objetivo, tiempo_estimado, int(duracion), profit, False, resultado)
                    manejar_resultado_contrato({"contract_id": contrato_id, "profit": profit})
                    break

            time.sleep(1)

    except Exception as e:
        log(f"[❌ ERROR GESTOR CONTRATO {contrato_id}] {e}", "error")
    finally:
        ws.close()
        CONTRACTS_ACTIVOS.pop(contrato_id, None)
