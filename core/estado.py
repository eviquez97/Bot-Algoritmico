# core/estado.py

# ==========================================
# 🌐 ESTADO GLOBAL DEL CONTRATO EN EJECUCIÓN
# ==========================================

# 🆔 Contrato abierto actualmente (ID único provisto por Deriv)
contrato_activo = None

# ⏱️ Timestamp UNIX (segundos) cuando se ejecutó el contrato
tiempo_inicio_contrato = None

# ⏳ Timestamp UNIX (segundos) del último chequeo de vigilancia
tiempo_vigilancia = None

# 💾 Último contrato registrado (respaldo para recuperación post-crash)
contrato_recuperado = None

# 🛰️ Indicador de que la vigilancia ya fue iniciada para el contrato actual
vigilancia_activada = False

# 📏 Duración estimada del contrato según predicción DRL
duracion_contrato_segundos = None

# 📊 Detalles de la operación actual en estructura única
datos_operacion = {
    "monto": None,                # 💵 Capital invertido
    "multiplicador": None,       # 🔁 Apalancamiento usado
    "score": None,               # 🎯 Score DRL asociado
    "prediccion_futura": None,   # 🔮 Predicción LSTM
    "ganancia_esperada": None,   # 💰 Valor objetivo que se espera alcanzar
    "duracion": None             # ⏱️ Duración estimada del contrato
}

# 🧠 Último resultado de predicción Spike IA (usado para evitar duplicar cálculos)
ultima_prediccion_spike = None

# 📉 Último tick recibido (usado por el recolector en tiempo real)
ultimo_tick = None

# ================================
# ✅ Función para actualizar estado
# ================================
def actualizar_estado_contrato(contrato_id, start_time):
    global contrato_activo, tiempo_inicio_contrato, vigilancia_activada, tiempo_vigilancia
    contrato_activo = contrato_id
    tiempo_inicio_contrato = start_time
    vigilancia_activada = False
    tiempo_vigilancia = None

# ================================
# ✅ Función para registrar tick
# ================================
def actualizar_tick(tick):
    global ultimo_tick
    ultimo_tick = tick


