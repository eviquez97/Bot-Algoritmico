# core/vigilancia_contrato.py

import time
import csv
import os
import pandas as pd
import core.estado as estado
from core.ia_spike import evaluar_spike_ia
from core.contexto import construir_contexto_para_spike
from core.cierre import cerrar_contrato_activo
from core.registro import registrar_resultado_contrato, guardar_experiencia_drl
from utils.logs import log
from utils.deriv_api import obtener_info_contrato_ws

RUTA_DATASET_DRL = "data/dataset_drl.csv"
RUTA_OPERATIVO = "data/dataset_operativo.csv"
RUTA_CSV = "data/contexto_historico.csv"
COLUMNAS_REQUERIDAS = ["close", "high", "low"]

def vigilar_contrato_con_websocket(contrato_id):
    try:
        if estado.vigilancia_activada:
            log("[⛔ VIGILANCIA WS] Ya hay un proceso de vigilancia activo.")
            return

        if not contrato_id:
            log("[⏳ WS] No hay contrato válido para monitorear.")
            return

        datos = estado.datos_operacion
        monto = datos.get("monto", 0)
        multiplicador = datos.get("multiplicador", 100)
        score_drl = datos.get("score", 0)
        prediccion_futura = datos.get("prediccion_futura", 0)
        duracion_estimada = datos.get("duracion", 60)
        ganancia_esperada = datos.get("ganancia_esperada", 1)

        estado.vigilancia_activada = True
        estado.contrato_activo = contrato_id

        log(f"[🔎 VIGILANCIA WS ACTIVADA] Contrato: {contrato_id}")
        tiempo_inicio = time.time()

        while True:
            time.sleep(2)
            tiempo_transcurrido = time.time() - tiempo_inicio

            info = obtener_info_contrato_ws(contrato_id)
            if not info:
                log("⚠️ [WS] No se pudo obtener info del contrato por WebSocket.")
                continue

            status = info.get("status")
            ganancia_actual = round(float(info.get("profit", 0.0)), 2)
            log(f"[⏱️ WS] {int(tiempo_transcurrido)}s | 💵 Ganancia: ${ganancia_actual:.2f}")

            if status == "sold":
                ganancia_final = _obtener_ganancia_final(contrato_id) or ganancia_actual
                log(f"🧨 [CIERRE WS] Contrato cerrado. Ganancia final: ${ganancia_final:.2f}")
                _cerrar_y_guardar_total(
                    contrato_id,
                    ganancia_final,
                    tiempo_transcurrido,
                    "🧨 Cierre por vencimiento",
                    score_drl,
                    prediccion_futura,
                    duracion_estimada,
                    ganancia_esperada,
                    monto,
                    multiplicador
                )
                break

            df_csv = pd.read_csv(RUTA_CSV, on_bad_lines="skip", engine="python")

            if df_csv.empty or df_csv.shape[0] < 130:
                log("[⏳ WS SPIKE] CSV insuficiente para Spike IA.")
                continue

            df_filtrado = df_csv.tail(150).iloc[:-1]
            if df_filtrado[COLUMNAS_REQUERIDAS].isnull().any().any():
                continue

            df_contexto = construir_contexto_para_spike(df_filtrado)
            if df_contexto is None or df_contexto.shape[0] < 30:
                continue

            pred = evaluar_spike_ia(df_contexto)
            if not pred or not isinstance(pred, dict):
                continue

            rf = pred.get("rf_spike", 0)
            lstm = pred.get("lstm_spike", 0)
            visual = pred.get("visual_spike", 0)

            log(f"[🧠 WS SPIKE V5] RF: {rf:.2f} | LSTM: {lstm:.2f} | Visual: {visual:.2f}")

            votos = sum([
                rf >= 0.60,
                lstm >= 0.40,
                visual >= 0.40
            ])

            if votos >= 2:
                log("🛡️ CIERRE ANTICIPADO WS: Spike IA detectado.")
                _cerrar_y_guardar_total(
                    contrato_id,
                    ganancia_actual,
                    tiempo_transcurrido,
                    "⚠️ Cierre anticipado por Spike IA",
                    score_drl,
                    prediccion_futura,
                    duracion_estimada,
                    ganancia_esperada,
                    monto,
                    multiplicador
                )
                break

    except Exception as e:
        log(f"[❌ ERROR VIGILANCIA WS] {e}")
    finally:
        estado.vigilancia_activada = False
        estado.contrato_activo = None

def _obtener_ganancia_final(contrato_id):
    info = obtener_info_contrato_ws(contrato_id)
    if info:
        return round(float(info.get("profit", 0.0)), 2)
    return 0.0

def _cerrar_y_guardar_total(contrato_id, ganancia, duracion, motivo, score, futuro, duracion_est, ganancia_esp, monto, mult):
    estado.ganancia_registrada = True
    registrar_resultado_contrato(ganancia, int(duracion), motivo)

    try:
        guardar_experiencia_drl(score, futuro, duracion_est, ganancia_esp, monto, mult, ganancia)

        with open(RUTA_DATASET_DRL, mode="a", newline="") as f:
            writer = csv.writer(f)
            if os.stat(RUTA_DATASET_DRL).st_size == 0:
                writer.writerow(["score", "futuro", "duracion_estim", "ganancia_esperada", "monto", "multiplicador", "resultado"])
            writer.writerow([score, futuro, duracion_est, ganancia_esp, monto, mult, ganancia])

        if os.path.exists(RUTA_OPERATIVO):
            df = pd.read_csv(RUTA_OPERATIVO)
            ultima_fila_idx = df.tail(1).index.item()
            df.loc[ultima_fila_idx, "ganancia_real"] = ganancia
            df.loc[ultima_fila_idx, "duracion_real"] = int(duracion)
            df.loc[ultima_fila_idx, "exito"] = 1 if ganancia > 0 else 0
            df.loc[ultima_fila_idx, "operacion_exitosa"] = 1
            df.to_csv(RUTA_OPERATIVO, index=False)
            log("✅ Dataset operativo actualizado con resultado real.")
        else:
            log("⚠️ No se encontró dataset_operativo.csv para registrar resultados.")

    except Exception as e:
        log(f"[❌ ERROR GUARDADO RESULTADOS] {e}")

    cerrar_contrato_activo()

