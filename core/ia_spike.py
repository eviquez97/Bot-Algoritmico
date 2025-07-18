# core/ia_spike.py

import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import joblib
from tensorflow.keras.models import load_model

from modelos.modelos_spike import model_spike, model_lstm_spike, scaler_rf_2, scaler_visual, scs_vision_x_model
from utils.logs import log
from core.contexto import construir_contexto_para_spike
from core.registro import registrar_spike_real

CSV_RESPALDO = "data/contexto_historico.csv"

@tf.function(reduce_retracing=True)
def predecir_lstm_spike(input_tensor):
    return model_lstm_spike(input_tensor, training=False)

def predecir_visual_spike(df):
    try:
        if df is None or df.shape[0] < 32:
            log("[❌ VISUAL SPIKE] Contexto insuficiente.")
            return 0.0

        columnas_visual = ['fuerza_cuerpo', 'fuerza_mecha', 'bajistas', 'rsi',
                           'momentum', 'spread', 'score', 'variacion', 'ema']
        df_visual = df[columnas_visual].tail(32).copy()

        if df_visual.shape[0] != 32:
            log(f"[❌ VISUAL SPIKE] Se requieren exactamente 32 filas, pero hay {df_visual.shape[0]}.")
            return 0.0

        contexto = scaler_visual.transform(df_visual.values)
        entrada = contexto.reshape(1, 32, 9)

        pred = scs_vision_x_model.predict(entrada, verbose=0)[0][0]
        return float(pred)
    except FileNotFoundError:
        log("[⚠️ VISUAL SPIKE] Modelo o scaler no disponible. Omitiendo predicción visual.")
        return 0.0
    except Exception as e:
        log(f"[❌ VISUAL SPIKE] {e}")
        return 0.0

def estimar_tiempo_spike(pred_lstm):
    if pred_lstm >= 0.85:
        return "0–1 minutos"
    elif pred_lstm >= 0.7:
        return "1–2 minutos"
    elif pred_lstm >= 0.55:
        return "2–3 minutos"
    else:
        return "3+ minutos"

def evaluar_spike_ia(df):
    try:
        df_contexto = construir_contexto_para_spike(df)
        respaldo_usado = False

        if df_contexto is None or len(df_contexto) < 30:
            log("[📂 CONTEXTO SPIKE] Insuficiente. Reintentando con respaldo CSV...")
            try:
                respaldo = pd.read_csv(CSV_RESPALDO)
                df_contexto = construir_contexto_para_spike(respaldo)
                respaldo_usado = True
            except Exception as e:
                log(f"[❌ ERROR LECTURA CSV RESPALDO] {e}")
                return {
                    "bloqueado": False,
                    "rf_spike": 0.0,
                    "lstm_spike": 0.0,
                    "visual_spike": 0.0,
                    "tiempo_estimado": "N/A",
                    "razones_bloqueo": []
                }

        if df_contexto is None or len(df_contexto) < 30:
            log("[⚠️ CONTEXTO SPIKE] Menos de 30 velas válidas. Spike IA omitido.")
            log("[🧠 SPIKE IA V5] RF: 0.00 | LSTM: 0.00 | Visual: 0.00")
            return {
                "bloqueado": False,
                "rf_spike": 0.0,
                "lstm_spike": 0.0,
                "visual_spike": 0.0,
                "tiempo_estimado": "N/A",
                "razones_bloqueo": []
            }

        columnas_rf_lstm = list(scaler_rf_2.feature_names_in_)
        df_rf_lstm = df_contexto[columnas_rf_lstm].tail(30).copy()

        log("[🔍 CONTEXTO SPIKE] Construido correctamente." + (" (CSV)" if respaldo_usado else ""))

        try:
            x_rf = scaler_rf_2.transform(df_rf_lstm)
            x_seq = np.reshape(x_rf, (1, 30, len(columnas_rf_lstm)))
        except Exception as e:
            log(f"[❌ ERROR SCALER SPIKE] {e}")
            return {
                "bloqueado": False,
                "rf_spike": 0.0,
                "lstm_spike": 0.0,
                "visual_spike": 0.0,
                "tiempo_estimado": "N/A",
                "razones_bloqueo": []
            }

        pred_rf, pred_lstm, pred_visual = 0.0, 0.0, 0.0
        try:
            pred_rf = float(model_spike.predict_proba(x_rf)[-1][1])
        except Exception as e:
            log(f"[❌ RF SPIKE] {e}")
        try:
            pred_lstm = float(predecir_lstm_spike(x_seq).numpy().flatten()[0])
        except Exception as e:
            log(f"[❌ LSTM SPIKE] {e}")
        try:
            pred_visual = float(predecir_visual_spike(df_contexto))
        except Exception as e:
            log(f"[❌ VISUAL SPIKE] {e}")

        log(f"[🧠 SPIKE IA V5] RF: {pred_rf:.2f} | LSTM: {pred_lstm:.2f} | Visual: {pred_visual:.2f}")

        if "close" in df.columns and "open" in df.columns:
            if df.iloc[-1]["close"] > df.iloc[-1]["open"]:
                log("🟢 SPIKE DETECTADO: Vela verde explosiva. Registrando...")
                try:
                    registrar_spike_real(df.iloc[-1].to_dict())
                except Exception as e:
                    log(f"[⚠️ ERROR REGISTRO SPIKE REAL] {e}")

        bloqueo = False
        razones = []

        if pred_rf >= 0.60 and pred_lstm >= 0.50:
            bloqueo = True
            razones.append("RF+LSTM")
        if pred_lstm >= 0.50 and pred_visual >= 0.50:
            bloqueo = True
            razones.append("LSTM+Visual")
        if pred_rf >= 0.60 and pred_visual >= 0.50:
            bloqueo = True
            razones.append("RF+Visual")

        if bloqueo:
            tiempo_estimado = estimar_tiempo_spike(pred_lstm)
            log(f"🛡️ BLOQUEO ACTIVADO: Spike anticipado detectado ({', '.join(razones)})")
            log(f"🕒 Spike estimado en aproximadamente {tiempo_estimado}.")
        else:
            log("🟢 No hay consenso de spike. Entrada permitida.")

        return {
            "bloqueado": bloqueo,
            "rf_spike": pred_rf,
            "lstm_spike": pred_lstm,
            "visual_spike": pred_visual,
            "tiempo_estimado": estimar_tiempo_spike(pred_lstm),
            "razones_bloqueo": razones
        }

    except Exception as e:
        log(f"[❌ ERROR SPIKE IA FATAL] {e}")
        log("[🧠 SPIKE IA V5] RF: 0.00 | LSTM: 0.00 | Visual: 0.00")
        return {
            "bloqueado": False,
            "rf_spike": 0.0,
            "lstm_spike": 0.0,
            "visual_spike": 0.0,
            "tiempo_estimado": "N/A",
            "razones_bloqueo": []
        }

# 🔁 Funciones auxiliares para test unitario y diagnóstico

def cargar_modelo_spike_rf():
    return joblib.load("modelos/model_spike.pkl")

def cargar_modelo_spike_lstm():
    return load_model("modelos/model_lstm_spike.keras")

def cargar_modelo_spike_visual():
    return load_model("modelos/scs_vision_x_model.keras")

def cargar_modelos_spike():
    try:
        log("[⚙️ CARGA] Cargando modelo Spike RF...")
        rf = cargar_modelo_spike_rf()
        log("[✅ MODELO RF] Cargado correctamente.")

        log("[⚙️ CARGA] Cargando modelo Spike LSTM...")
        lstm = cargar_modelo_spike_lstm()
        log("[✅ MODELO LSTM] Cargado correctamente.")

        log("[⚙️ CARGA] Cargando modelo Spike Visual...")
        visual = cargar_modelo_spike_visual()
        log("[✅ MODELO VISUAL] Cargado correctamente.")

        return {
            "rf": rf,
            "lstm": lstm,
            "visual": visual
        }

    except Exception as e:
        log(f"[❌ ERROR AL CARGAR MODELOS SPIKE] {e}")
        return None
