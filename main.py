import time
import pandas as pd
import threading
import os
from datetime import datetime, timedelta

# Logs
from utils.logs import log

# Entrenamiento en vivo
from core.drl_adaptativo import evaluar_rendimiento_drl
from core.ia_drl_entrenamiento import entrenar_drl_en_vivo

# WebSocket y vigilancia
from core.ws_ticks import iniciar_ws_ticks
from core.ws_verificacion import iniciar_verificacion_contratos
from core.ws_vigilancia_contrato import iniciar_vigilancia_contratos_activos
from core.watchdog import iniciar_watchdog

# IA y buffers
from core.buffer import VELAS_BUFFER
from core.gestion_diaria import cargar_estado_diario
from core.autocuracion import verificar_autocuracion
from core.ia_spike import evaluar_spike_ia
from core.ia_modelos import model_scs_vision_x as scs_vision_x_model
from core.verificador_drl import verificar_integridad_drl
from core.verificador_cierre import verificar_integridad_cierre
from core.spike_guardian import verificar_reentrenamiento_spike

# ENTRENAMIENTOS
from reentrenamiento.entrenador_cierre import entrenar_modelo_cierre
from reentrenamiento.autoentrenador import verificar_reentrenamiento_general
from reentrenamiento.verificador_spike import verificar_integridad_spike
from reentrenamiento.entrenador_drl import entrenar_modelo_drl
from reentrenamiento.entrenador_futuro import entrenar as entrenar_modelo_futuro
from reentrenamiento.entrenador_spike import entrenar_modelo_spike

# Modelos adicionales
from core.ia_ganancia import cargar_modelo_ganancia

# TensorFlow silencioso
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# FLAGS y rutas
MODO_REAL = True
CSV_CONTEXTO = "data/contexto_historico.csv"
EXPERIENCIAS_DRL = "data/experiencias_drl.csv"
RUTA_REENTRENAMIENTO_CIERRE = "logs/ultima_actualizacion_cierre.txt"
RUTA_REENTRENAMIENTO_FUTURO = "logs/ultima_actualizacion_futuro.txt"

# 🔁 Ciclo de entrenamiento DRL
def ciclo_entrenamiento_drl():
    while True:
        try:
            if os.path.exists(EXPERIENCIAS_DRL):
                df_drl = pd.read_csv(EXPERIENCIAS_DRL)
                if len(df_drl.columns) != 12:
                    log("❌ [DRL] Dataset inválido: columnas incorrectas")
                elif len(df_drl) >= 30:
                    log("🔁 Reentrenamiento DRL iniciado.")
                    entrenar_modelo_drl()
                    log("✅ Reentrenamiento DRL completado.")
                else:
                    log("[⛔ DRL] Dataset con menos de 30 filas. Entrenamiento omitido.")
            else:
                log("[⛔ DRL] Dataset no encontrado.")
        except Exception as e:
            log(f"[❌ ERROR ENTRENAMIENTO DRL] {e}")
        time.sleep(3600)

# 🔁 Reentrenamiento de cierre cada 12h
def verificar_reentrenamiento_cierre():
    try:
        ahora = datetime.now()
        if os.path.exists(RUTA_REENTRENAMIENTO_CIERRE):
            with open(RUTA_REENTRENAMIENTO_CIERRE, "r") as f:
                ultima = datetime.strptime(f.read().strip(), "%Y-%m-%d %H:%M:%S")
        else:
            ultima = ahora - timedelta(hours=13)

        if ahora - ultima >= timedelta(hours=12):
            log("🔁 [AUTO CIERRE] Reentrenando modelo de cierre...")
            entrenar_modelo_cierre()
            with open(RUTA_REENTRENAMIENTO_CIERRE, "w") as f:
                f.write(ahora.strftime("%Y-%m-%d %H:%M:%S"))
            log("✅ [AUTO CIERRE] Completado.")
        else:
            faltan = timedelta(hours=12) - (ahora - ultima)
            log(f"[⏳ AUTO CIERRE] Próximo en: {faltan}")
    except Exception as e:
        log(f"[❌ ERROR AUTO CIERRE] {e}")

# 🔁 Reentrenamiento futuro cada 30min
def verificar_reentrenamiento_futuro():
    try:
        ahora = datetime.now()
        if os.path.exists(RUTA_REENTRENAMIENTO_FUTURO):
            with open(RUTA_REENTRENAMIENTO_FUTURO, "r") as f:
                ultima = datetime.strptime(f.read().strip(), "%Y-%m-%d %H:%M:%S")
        else:
            ultima = ahora - timedelta(hours=2)

        if ahora - ultima >= timedelta(minutes=30):
            log("🔁 [AUTO FUTURO] Reentrenando modelo de predicción futura...")
            entrenar_modelo_futuro()
            with open(RUTA_REENTRENAMIENTO_FUTURO, "w") as f:
                f.write(ahora.strftime("%Y-%m-%d %H:%M:%S"))
            log("✅ [AUTO FUTURO] Completado.")
        else:
            faltan = timedelta(minutes=30) - (ahora - ultima)
            log(f"[⏳ AUTO FUTURO] Próximo chequeo en: {faltan}")
    except Exception as e:
        log(f"[❌ ERROR AUTO FUTURO] {e}")

# 📂 Carga de velas desde CSV
def cargar_velas_desde_csv():
    try:
        df = pd.read_csv(CSV_CONTEXTO)
        if len(df) >= 120:
            velas = df.tail(120).to_dict(orient="records")
            log(f"[📂 CSV] Cargadas {len(velas)} velas.")
            return velas
        else:
            log(f"[⚠️ CSV] Solo hay {len(df)} velas. Se requieren mínimo 120.")
            return []
    except FileNotFoundError:
        log("[⚠️ CSV] Archivo no encontrado.")
        return []
    except Exception as e:
        log(f"[❌ ERROR CSV] {e}")
        return []

# 🚀 INICIO
if __name__ == "__main__":
    modo = "REAL" if MODO_REAL else "DEMO"
    log(f"🧠 BOT MONSTRUO ✅ INICIADO | MODO: {modo} | BOOM1000 | Entrenamiento adaptativo activo")

    cargar_estado_diario()
    VELAS_BUFFER.extend(cargar_velas_desde_csv())

    # ✅ Evaluación inicial de SPIKE solo si hay suficientes velas válidas
    if len(VELAS_BUFFER) >= 60:
        df_inicial = pd.DataFrame(VELAS_BUFFER[-60:])
        if all(col in df_inicial.columns for col in ["open", "high", "low", "close"]):
            try:
                evaluar_spike_ia(df_inicial)
            except Exception as e:
                log(f"[❌ ERROR SPIKE IA INICIAL] {e}")
        else:
            log("[❌ ERROR SPIKE IA] Velas cargadas no tienen columnas mínimas requeridas.")
    else:
        log("[⚠️ SPIKE IA OMITIDO] No hay suficientes velas para evaluación inicial.")

    iniciar_ws_ticks()
    iniciar_verificacion_contratos()
    iniciar_vigilancia_contratos_activos()
    iniciar_watchdog()
    threading.Thread(target=ciclo_entrenamiento_drl, daemon=True).start()

    while True:
        try:
            evaluar_rendimiento_drl()
            verificar_integridad_drl()
            verificar_integridad_cierre()
            verificar_reentrenamiento_general()
            verificar_reentrenamiento_spike()
            verificar_reentrenamiento_cierre()
            verificar_reentrenamiento_futuro()
            verificar_integridad_spike()
            verificar_autocuracion()
            time.sleep(60)
        except Exception as e:
            log(f"[❌ ERROR EN BUCLE PRINCIPAL] {e}")
            time.sleep(10)
