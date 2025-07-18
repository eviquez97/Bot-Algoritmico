# core/modelo.py

from core.ia_spike import cargar_modelo_spike_rf, cargar_modelo_spike_lstm, cargar_modelo_spike_visual
from core.ia_drl import cargar_modelo_drl_direccion, cargar_modelo_drl_ganancia
from core.ia_cierre import cargar_modelo_scpx
from utils.logs import log

def cargar_modelos_spike():
    try:
        modelo_rf = cargar_modelo_spike_rf()
        modelo_lstm = cargar_modelo_spike_lstm()
        modelo_visual = cargar_modelo_spike_visual()
        log("✅ Modelos Spike IA cargados correctamente.")
        return {
            "rf": modelo_rf,
            "lstm": modelo_lstm,
            "visual": modelo_visual
        }
    except Exception as e:
        log(f"[❌ ERROR AL CARGAR MODELOS SPIKE] {e}")
        return {}

def cargar_modelos_drl():
    try:
        modelo_direccion = cargar_modelo_drl_direccion()
        modelo_ganancia = cargar_modelo_drl_ganancia()
        log("✅ Modelos DRL cargados correctamente.")
        return {
            "direccion": modelo_direccion,
            "ganancia": modelo_ganancia
        }
    except Exception as e:
        log(f"[❌ ERROR AL CARGAR MODELOS DRL] {e}")
        return {}

def cargar_modelo_scpx():
    try:
        modelo = cargar_modelo_scpx()
        log("[📥 SCDP-X] Modelo y scaler de cierre cargados correctamente.")
        return modelo
    except Exception as e:
        log(f"[❌ ERROR AL CARGAR MODELO SCDP-X] {e}")
        return None
