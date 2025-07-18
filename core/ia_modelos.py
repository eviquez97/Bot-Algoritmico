# core/ia_modelos.py
import os
import joblib
import tensorflow as tf
from utils.logs import log

# =============================
# Carga del modelo DRL
# =============================
MODELO_DRL_PATH = "modelos/modelo_drl.keras"  # ✅ Nombre corregido
modelo_drl = None

if os.path.exists(MODELO_DRL_PATH):
    try:
        modelo_drl = tf.keras.models.load_model(MODELO_DRL_PATH)
        log("✅ Modelo DRL cargado exitosamente.")
    except Exception as e:
        log(f"[❌ ERROR DRL] No se pudo cargar el modelo DRL: {e}")
        modelo_drl = None
else:
    log(f"[❌ ERROR] No se encontró el modelo DRL en {MODELO_DRL_PATH}")
    modelo_drl = None

# =============================
# Carga del modelo Spike Visual (SCS-VISION X)
# =============================
MODELO_SCS_PATH = "modelos/model_scs_vision_x.keras"
model_scs_vision_x = None

if os.path.exists(MODELO_SCS_PATH):
    try:
        model_scs_vision_x = tf.keras.models.load_model(MODELO_SCS_PATH)
        log("✅ Modelo Spike Visual (SCS-VISION X) cargado exitosamente.")
    except Exception as e:
        log(f"[❌ ERROR SPIKE VISUAL] No se pudo cargar el modelo visual: {e}")
else:
    log(f"[❌ ERROR] No se encontró el modelo Spike Visual en {MODELO_SCS_PATH}")

# =============================
# Carga modelos Random Forest
# =============================
try:
    modelo_rf_ganancia = joblib.load("modelos/model_rf_ganancia.pkl")
    log("✅ Modelo RF de Ganancia cargado correctamente.")
except Exception as e:
    modelo_rf_ganancia = None
    log(f"[❌ ERROR RF GANANCIA] {e}")

try:
    modelo_rf_direccion = joblib.load("modelos/model_rf_direccion.pkl")
    log("✅ Modelo RF de Dirección cargado correctamente.")
except Exception as e:
    modelo_rf_direccion = None
    log(f"[❌ ERROR RF DIRECCIÓN] {e}")
