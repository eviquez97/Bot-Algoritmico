# registro.py

import os
import csv
from datetime import datetime, date
import pandas as pd
from utils.logs import log

# Rutas
EXPERIENCIAS_DRL = "data/experiencias_drl.csv"
DATASET_DRL = "data/dataset_drl.csv"
RESULTADOS_DRL = "data/resultados_drl.csv"
RESULTADOS_CIERRE = "data/resultados_cierre.csv"
REGISTRO_GANANCIAS_DIARIAS = "data/ganancias_diarias.csv"
REGISTRO_SPIKE_SIMPLE = "data/spikes_detectados.csv"
DATASET_SPIKE_MODELO = "data/dataset_operativo.csv"

META_DIARIA = 500.0


def escribir_fila_con_columnas_validadas(archivo, fila, columnas_esperadas):
    archivo_existe = os.path.exists(archivo)
    fila_completa = {col: fila.get(col, 0) for col in columnas_esperadas}
    with open(archivo, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columnas_esperadas)
        if not archivo_existe or os.path.getsize(archivo) == 0:
            writer.writeheader()
        writer.writerow(fila_completa)


def registrar_experiencia_drl(timestamp, score, ganancia_estim, porcentaje_bajistas,
                               pred_futuro, ema, rsi, momentum, spread,
                               pred_rf, pred_lstm, pred_visual, ultima_direccion,
                               accion, monto, multiplicador, exito):
    try:
        fila = {
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "score": float(score or 0),
            "ganancia_estimada": float(ganancia_estim or 0),
            "porcentaje_bajistas": float(porcentaje_bajistas or 0),
            "pred_futuro": float(pred_futuro or 0),
            "ema": float(ema or 0),
            "rsi": float(rsi or 0),
            "momentum": float(momentum or 0),
            "spread": float(spread or 0),
            "pred_rf": float(pred_rf or 0),
            "pred_lstm": float(pred_lstm or 0),
            "pred_visual": float(pred_visual or 0),
            "ultima_direccion": ultima_direccion,
            "accion": int(accion or 0),
            "monto": float(monto or 1),
            "multiplicador": int(multiplicador or 100),
            "exito": int(exito or 0)
        }

        columnas = list(fila.keys())
        escribir_fila_con_columnas_validadas(EXPERIENCIAS_DRL, fila, columnas)
        escribir_fila_con_columnas_validadas(DATASET_DRL, fila, columnas)

        log("[🧠 DRL] Experiencia registrada correctamente.")
    except Exception as e:
        log(f"[❌ ERROR REGISTRO DRL] {e}")


def guardar_experiencia_drl(score, futuro, duracion, ganancia_esperada, monto, mult, ganancia_real):
    try:
        fila = {
            "score_drl": float(score or 0),
            "prediccion_futura": float(futuro or 0),
            "duracion_estimada": int(duracion or 0),
            "ganancia_esperada": float(ganancia_esperada or 0),
            "monto": float(monto or 0),
            "multiplicador": int(mult or 100),
            "ganancia_real": float(ganancia_real or 0)
        }
        columnas = list(fila.keys())
        escribir_fila_con_columnas_validadas(EXPERIENCIAS_DRL, fila, columnas)
        log("[📘 EXPERIENCIA DRL] Datos reales guardados.")
    except Exception as e:
        log(f"[❌ ERROR GUARDADO EXPERIENCIA DRL] {e}")


def registrar_resultado_contrato(ganancia, duracion, exito):
    try:
        from core.estado import datos_operacion

        columnas = [
            "timestamp", "score_drl", "monto", "multiplicador",
            "prediccion_futura", "ganancia_esperada", "duracion_estimada",
            "ganancia_real", "operacion_exitosa"
        ]

        fila = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "score_drl": float(datos_operacion.get("score", 0)),
            "monto": float(datos_operacion.get("monto", 0)),
            "multiplicador": int(datos_operacion.get("multiplicador", 100)),
            "prediccion_futura": float(datos_operacion.get("prediccion_futura", 0)),
            "ganancia_esperada": float(datos_operacion.get("ganancia_esperada", 0)),
            "duracion_estimada": int(datos_operacion.get("duracion", 0)),
            "ganancia_real": float(ganancia),
            "operacion_exitosa": int(exito)
        }

        escribir_fila_con_columnas_validadas(DATASET_SPIKE_MODELO, fila, columnas)

        log(f"[📈 DRL CONTRATO] Resultado real registrado: ${ganancia:.2f} | Duración: {duracion}s | Exito={exito}")
        registrar_ganancia(ganancia)

    except Exception as e:
        log(f"[❌ ERROR REGISTRO RESULTADO DRL] {e}")


def registrar_ganancia(monto):
    try:
        hoy = date.today().strftime("%Y-%m-%d")
        if not os.path.exists(REGISTRO_GANANCIAS_DIARIAS):
            with open(REGISTRO_GANANCIAS_DIARIAS, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["fecha", "ganancia_total"])

        df = pd.read_csv(REGISTRO_GANANCIAS_DIARIAS)
        if hoy in df["fecha"].values:
            df.loc[df["fecha"] == hoy, "ganancia_total"] += float(monto)
        else:
            nueva_fila = pd.DataFrame([{"fecha": hoy, "ganancia_total": float(monto)}])
            df = pd.concat([df, nueva_fila], ignore_index=True)

        df.to_csv(REGISTRO_GANANCIAS_DIARIAS, index=False)
        log(f"[💰 GANANCIA] +${monto:.2f} | Total hoy: ${df[df['fecha'] == hoy]['ganancia_total'].values[0]:.2f}")
    except Exception as e:
        log(f"[❌ ERROR GANANCIA] {e}")


def objetivo_diario_alcanzado():
    try:
        hoy = date.today().strftime("%Y-%m-%d")
        if not os.path.exists(REGISTRO_GANANCIAS_DIARIAS):
            return False
        df = pd.read_csv(REGISTRO_GANANCIAS_DIARIAS)
        fila = df[df["fecha"] == hoy]
        if not fila.empty and fila["ganancia_total"].values[0] >= META_DIARIA:
            log(f"[🏁 META ALCANZADA] Se ha alcanzado la meta diaria de ${META_DIARIA}")
            return True
        return False
    except Exception as e:
        log(f"[❌ ERROR META DIARIA] {e}")
        return False


def registrar_resultado_cierre(df, decision):
    try:
        df = df.copy()
        df["decision_cierre"] = decision
        df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        columnas = list(df.columns)
        escribir_fila_con_columnas_validadas(RESULTADOS_CIERRE, df.iloc[0].to_dict(), columnas)
    except Exception as e:
        log(f"[❌ ERROR REGISTRO CIERRE] {e}")


def registrar_spike_real(vela):
    try:
        columnas_simple = ['timestamp', 'epoch', 'open', 'high', 'low', 'close']
        fila_simple = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'epoch': vela.get("epoch", ""),
            'open': vela.get("open", 0),
            'high': vela.get("high", 0),
            'low': vela.get("low", 0),
            'close': vela.get("close", 0)
        }
        escribir_fila_con_columnas_validadas(REGISTRO_SPIKE_SIMPLE, fila_simple, columnas_simple)

        columnas_modelo = [
            'timestamp', 'fecha', 'hora', 'open', 'high', 'low', 'close', 'spread',
            'fuerza_cuerpo', 'fuerza_mecha', 'score', 'rsi', 'ema', 'momentum',
            'variacion', 'alcista', 'volumen_tick', 'spike_real', 'spike_anticipado',
            'ganancia_esperada', 'duracion_estimada', 'operacion_exitosa'
        ]

        fila_modelo = {col: vela.get(col, 0) for col in columnas_modelo}
        escribir_fila_con_columnas_validadas(DATASET_SPIKE_MODELO, fila_modelo, columnas_modelo)

        log(f"[🧠 SPIKE REGISTRADO] Epoch={fila_simple['epoch']}")
    except Exception as e:
        log(f"[❌ ERROR REGISTRO SPIKE] {e}")
