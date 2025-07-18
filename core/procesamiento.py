# core/procesamiento.py

import pandas as pd
import numpy as np
import os
import csv
import time
from core.contexto import construir_contexto
from core.filtros import es_entrada_necesaria
from core.ia_drl import procesar_decision_drl
from core.ia_cierre import evaluar_scpx
from core.ia_spike import evaluar_spike_ia
from core.buffer import VELAS_BUFFER
from core.estado import contrato_activo, datos_operacion
from core.registro import registrar_spike_real
from utils.logs import log
from core.operaciones import ejecutar_operacion_put

ya_imprimio_mensaje = False
CSV_CONTEXTO = "data/contexto_historico.csv"
CSV_SPIKE = "data/dataset_operativo.csv"

# Precarga
precargado = False
for ruta in [CSV_CONTEXTO, CSV_SPIKE]:
    if os.path.exists(ruta):
        try:
            df_csv = pd.read_csv(ruta)
            if len(df_csv) >= 1:
                log(f"[⚠️ PRECARGA] Precargando buffer desde {ruta} ({len(df_csv)} velas)...")
                for _, fila in df_csv.tail(200).iterrows():
                    VELAS_BUFFER.append(fila.to_dict())
                precargado = True
                break
        except Exception as e:
            log(f"[❌ ERROR PRECARGA {ruta}] {e}")

if not precargado:
    log("[⚠️ SIN PRECARGA] No se encontraron archivos válidos para precargar el buffer.")

def calcular_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean().replace(0, 1e-10)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.replace([np.inf, -np.inf], 50).fillna(50)

def procesar_vela(vela):
    global ya_imprimio_mensaje

    if not vela or any(pd.isna(vela.get(col)) for col in ["open", "high", "low", "close"]):
        log("[⚠️ VELA OMITIDA] Vela inválida o con NaNs.")
        return

    vela["timestamp"] = int(time.time())
    vela["volumen_tick"] = vela.get("volumen_tick", 0)
    VELAS_BUFFER.append(vela)
    log(f"[🕒 VELA PROCESADA] Añadida al buffer | Total: {len(VELAS_BUFFER)}/60")

    if len(VELAS_BUFFER) < 120:
        if not ya_imprimio_mensaje:
            log(f"[⏳ ESPERA] Aún no hay suficientes velas para análisis ({len(VELAS_BUFFER)}/120)")
            ya_imprimio_mensaje = True
        return

    try:
        df = pd.DataFrame(VELAS_BUFFER[-120:])
        df["spread"] = df["high"] - df["low"]
        df["momentum"] = df["close"].diff()
        df["variacion"] = (df["close"] - df["open"]) / df["open"]
        df["score"] = df["variacion"].rolling(window=5).mean()
        df["rsi"] = calcular_rsi(df["close"])
        df["ema"] = df["close"].ewm(span=10).mean()
        df["fuerza_cuerpo"] = abs(df["close"] - df["open"])
        df["mecha_superior"] = df["high"] - df[["close", "open"]].max(axis=1)
        df["mecha_inferior"] = df[["close", "open"]].min(axis=1) - df["low"]
        df["fuerza_mecha"] = df["mecha_superior"] + df["mecha_inferior"]
        df["bajistas"] = (df["close"] < df["open"]).astype(int)
        df["spike_real"] = (df["close"] > df["open"]).astype(int)
        df["spike_anticipado"] = 0

        for i in df[df["spike_real"] == 1].index:
            anticipadas = range(max(0, i - 3), i)
            df.loc[anticipadas, "spike_anticipado"] = 1

        df_crudo = df.copy()
        df = df.dropna(subset=["open", "high", "low", "close", "spread", "momentum", "variacion", "score", "rsi", "ema"])
        df = df.tail(60)

        evaluar_scpx(df)

        resultado_spike = evaluar_spike_ia(df)
        log(f"[🧠 SPIKE IA V5] RF: {resultado_spike.get('rf_spike', 0.0):.2f} | "
            f"LSTM: {resultado_spike.get('lstm_spike', 0.0):.2f} | "
            f"Visual: {resultado_spike.get('visual_spike', 0.0):.2f}")
        if resultado_spike.get("bloqueado", False):
            log("🛑 Entrada bloqueada por spike IA.")
            return

        if contrato_activo:
            log("[🔒 BLOQUEO] Hay un contrato activo en vigilancia. Solo se analiza, no se ejecuta.")
            return

        contexto = construir_contexto(df_crudo, cantidad=60)
        if contexto is None:
            log("[❌ CONTEXTO] Error al construir contexto. Operación omitida.")
            return

        capital = 500.0
        multiplicadores = [100, 200, 300, 400]
        decision = procesar_decision_drl(contexto, capital, multiplicadores)

        score_drl = decision.get("score", 0.0)
        accion = decision.get("accion", 0)
        monto = decision.get("monto", 0)
        mult = decision.get("multiplicador", 0)
        ganancia = decision.get("ganancia_esperada", 0.0)
        duracion = decision.get("duracion_estimada", 0)
        prediccion_futura = decision.get("prediccion_futura", 0.0)
        permitir = decision.get("permitir_entrada", False)

        fila_cruda = df.iloc[-1].copy()
        fila_final = {
            "timestamp": int(time.time()),
            "open": float(fila_cruda.get("open", 0)),
            "high": float(fila_cruda.get("high", 0)),
            "low": float(fila_cruda.get("low", 0)),
            "close": float(fila_cruda.get("close", 0)),
            "spread": float(fila_cruda.get("spread", 0)),
            "momentum": float(fila_cruda.get("momentum", 0)),
            "variacion": float(fila_cruda.get("variacion", 0)),
            "score": float(fila_cruda.get("score", 0)),
            "rsi": float(fila_cruda.get("rsi", 0)),
            "ema": float(fila_cruda.get("ema", 0)),
            "fuerza_cuerpo": float(fila_cruda.get("fuerza_cuerpo", 0)),
            "mecha_superior": float(fila_cruda.get("mecha_superior", 0)),
            "mecha_inferior": float(fila_cruda.get("mecha_inferior", 0)),
            "fuerza_mecha": float(fila_cruda.get("fuerza_mecha", 0)),
            "bajistas": int(fila_cruda.get("bajistas", 0)),
            "rf_spike": float(resultado_spike.get("rf_spike", 0.0)),
            "lstm_spike": float(resultado_spike.get("lstm_spike", 0.0)),
            "visual_spike": float(resultado_spike.get("visual_spike", 0.0)),
            "prediccion_futura": float(prediccion_futura),
            "score_drl": float(score_drl),
            "accion": int(accion),
            "monto": float(monto),
            "multiplicador": float(mult),
            "ganancia_esperada": float(ganancia),
            "duracion_estimada": int(duracion),
            "ganancia_real": None,
            "duracion_real": None,
            "exito": None,
            "operacion_exitosa": None,
            "ultima_direccion": 1 if fila_cruda.get("close", 0) > fila_cruda.get("open", 0) else 0,
        }

        columnas = list(fila_final.keys())
        for archivo in [CSV_SPIKE, CSV_CONTEXTO]:
            os.makedirs(os.path.dirname(archivo), exist_ok=True)
            escribir_header = not os.path.exists(archivo) or os.path.getsize(archivo) == 0
            with open(archivo, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=columnas)
                if escribir_header:
                    writer.writeheader()
                writer.writerow(fila_final)
        log(f"[💾 CSV] Fila guardada en {CSV_SPIKE}")

        if fila_cruda.get("close", 0) > fila_cruda.get("open", 0):
            registrar_spike_real(fila_final)
            log("🟢 SPIKE DETECTADO: Vela verde explosiva.")

        if permitir and monto > 0 and mult > 0:
            log(f"[🚦 ENTRADA AUTORIZADA] Ejecutando operación real: ${monto} x{mult}")
            ejecutar_operacion_put(monto, mult, score_drl, prediccion_futura, ganancia, duracion)
        else:
            log("[⚠️ MODO APRENDIZAJE] Ejecutando operación forzada para alimentar dataset.")
            ejecutar_operacion_put(1.0, 100, score_drl, prediccion_futura, 1.5, 120)

    except Exception as e:
        log(f"[❌ ERROR PROCESAMIENTO] {e}")

