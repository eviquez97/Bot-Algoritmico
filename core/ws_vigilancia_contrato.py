# core/ws_vigilancia_contrato.py

import threading
import time
import pandas as pd
from utils.logs import log
from utils.deriv_api import obtener_info_contrato
from core.estado import contrato_activo, datos_operacion
from core.cierre import cerrar_contrato_activo
from core.contexto import construir_contexto_para_spike
from core.ia_spike import evaluar_spike_ia
from core.registro import registrar_resultado_contrato, guardar_experiencia_drl

RUTA_CSV = "data/contexto_historico.csv"
RUTA_DATASET_DRL = "data/dataset_drl.csv"

INTERVALO_REVISION = 2
UMBRAL_RF = 0.60
UMBRAL_LSTM = 0.40
UMBRAL_VISUAL = 0.40

COLUMNAS_REQUERIDAS = ["close", "high", "low"]

def monitorear_contratos_activos():
    while True:
        try:
            if contrato_activo:
                info = obtener_info_contrato(contrato_activo)
                if info:
                    if info.get("is_sold", False):
                        log(f"✅ [WS VIGILANCIA] Contrato cerrado en Deriv: {contrato_activo}")
                        profit = info.get("profit", 0.0)
                        duracion = info.get("duration", 0)
                        registrar_resultado_contrato(profit, duracion, "🛑 Detectado cierre externo")
                        guardar_experiencia_drl(**datos_operacion, ganancia_obtenida=profit)
                        cerrar_contrato_activo()
                        continue
                    else:
                        log(f"[👁️ WS VIGILANCIA] Contrato sigue abierto en Deriv: {contrato_activo}")
                else:
                    log("[⚠️ WS VIGILANCIA] No se pudo obtener info del contrato.")

                # SPIKE IA – CIERRE ANTICIPADO
                try:
                    df_csv = pd.read_csv(RUTA_CSV, on_bad_lines="skip", engine="python")
                    if df_csv.empty or df_csv.shape[0] < 130:
                        log("[⏳ CONTEXTO SPIKE] CSV vacío o insuficiente.")
                        continue

                    df_filtrado = df_csv.tail(150).iloc[:-1]
                    if df_filtrado[COLUMNAS_REQUERIDAS].isnull().any().any():
                        log("[❌ CONTEXTO SPIKE] Columnas clave con NaN.")
                        continue

                    df_contexto = construir_contexto_para_spike(df_filtrado)
                    if df_contexto is None or df_contexto.shape[0] < 30:
                        log("[❌ CONTEXTO SPIKE] Contexto inválido.")
                        continue

                    pred = evaluar_spike_ia(df_contexto)
                    rf = pred.get("rf_spike", 0)
                    lstm = pred.get("lstm_spike", 0)
                    visual = pred.get("visual_spike", 0)

                    log(f"[🧠 WS SPIKE] RF: {rf:.2f} | LSTM: {lstm:.2f} | Visual: {visual:.2f}")

                    votos = sum([
                        rf >= UMBRAL_RF,
                        lstm >= UMBRAL_LSTM,
                        visual >= UMBRAL_VISUAL
                    ])

                    if votos >= 2:
                        log("🛡️ CIERRE ANTICIPADO: Spike IA detectado.")
                        profit = info.get("profit", 0.0)
                        duracion = info.get("duration", 0)

                        registrar_resultado_contrato(profit, duracion, "⚠️ Cierre anticipado Spike IA")
                        guardar_experiencia_drl(**datos_operacion, ganancia_obtenida=profit)
                        cerrar_contrato_activo()
                        continue

                except Exception as e:
                    log(f"[❌ ERROR SPIKE WS] {e}")

        except Exception as e:
            log(f"[❌ ERROR WS VIGILANCIA] {e}")

        time.sleep(INTERVALO_REVISION)

def iniciar_vigilancia_contratos_activos():
    hilo = threading.Thread(target=monitorear_contratos_activos, daemon=True)
    hilo.start()
    log("[🧠 WS VIGILANCIA] Hilo de vigilancia de contratos iniciado.")

