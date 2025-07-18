# test_inicio_diagnostico.py

import traceback
from utils.logs import log

print("\n🔍 INICIO DE DIAGNÓSTICO DE MODELOS IA...\n")

# ============================
# SPIKE IA - Random Forest
# ============================
try:
    from core.ia_spike import cargar_modelo_spike_rf
    modelo_rf = cargar_modelo_spike_rf()
    print("✅ RF Spike cargado correctamente.")
except Exception as e:
    print("❌ Error al cargar RF Spike:")
    traceback.print_exc()

# ============================
# SPIKE IA - LSTM
# ============================
try:
    from core.ia_spike import cargar_modelo_spike_lstm
    modelo_lstm = cargar_modelo_spike_lstm()
    print("✅ LSTM Spike cargado correctamente.")
except Exception as e:
    print("❌ Error al cargar LSTM Spike:")
    traceback.print_exc()

# ============================
# SPIKE IA - Visual CNN
# ============================
try:
    from core.ia_spike import cargar_modelo_spike_visual
    modelo_visual = cargar_modelo_spike_visual()
    print("✅ Visual Spike cargado correctamente.")
except Exception as e:
    print("❌ Error al cargar Visual Spike:")
    traceback.print_exc()

# ============================
# DRL IA - Deep Q-Learning
# ============================
try:
    from core.modelos_drl import modelo_drl
    if modelo_drl:
        print("✅ Modelo DRL cargado correctamente.")
    else:
        print("⚠️  Modelo DRL está vacío o no inicializado.")
except Exception as e:
    print("❌ Error al cargar modelo DRL:")
    traceback.print_exc()

# ============================
# SCDP-X - Cierre por IA
# ============================
try:
    from core.ia_cierre import cargar_modelo_cierre
    cargar_modelo_cierre()
    print("✅ Modelo de cierre (SCDP-X) cargado correctamente.")
except Exception as e:
    print("❌ Error al cargar modelo de cierre SCDP-X:")
    traceback.print_exc()

print("\n✅ Diagnóstico finalizado.\n")
