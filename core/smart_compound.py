# core/smart_compound.py

import os
from core.estado import contrato_activo, datos_operacion
from core.registro import objetivo_diario_alcanzado
from utils.logs import log

# 🧮 Interés compuesto dinámico con capital recibido
def calcular_monto_multiplicador(capital_actual, score):
    base = capital_actual * 0.05
    riesgo = min(max(score / 100, 0.01), 0.20)
    monto = round(base * (1 + riesgo), 2)

    if monto <= 0.0:
        log("[⚠️ MONTO CERO] Capital muy bajo. Se fuerza monto mínimo de $5")
        monto = 5.0

    return monto

# 🧠 Determina multiplicador por predicción futura
def calcular_multiplicador(pred_futuro):
    if pred_futuro > 0.9:
        return 400
    elif pred_futuro > 0.8:
        return 300
    elif pred_futuro > 0.7:
        return 200
    else:
        return 100

# 🎯 Versión compacta para DRL
def obtener_entrada_dinamica(capital_actual, score, pred_futuro):
    monto = calcular_monto_multiplicador(capital_actual, score)
    multiplicador = calcular_multiplicador(pred_futuro)
    return monto, multiplicador

# 💣 Evaluación SCM para cierre anticipado por ganancia parcial
def evaluar_scm(df):
    if contrato_activo is None:
        return False

    try:
        actual = df.iloc[-1]
        apertura = df.iloc[-2] if len(df) >= 2 else actual

        ganancia_obj = datos_operacion.get("ganancia_esperada", 0)
        precio_apertura = apertura["close"]
        precio_actual = actual["close"]

        if precio_apertura == 0:
            return False

        cambio = (precio_apertura - precio_actual) / precio_apertura

        if cambio >= 0.7 * ganancia_obj:
            log(f"[📈 SCM] Cierre anticipado: Ganancia parcial alcanzada ({cambio:.2f})")
            return True

    except Exception as e:
        log(f"[❌ ERROR SCM] {e}")

    return False

