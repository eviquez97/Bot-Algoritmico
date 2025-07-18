import pandas as pd
from core.spike_guardian_predictivo import evaluar_spike_anticipado
from core.drl import procesar_decision_drl
from utils.log import log
from datetime import datetime
from utils.contexto import construir_contexto_drl

# === BUFFER DE VELAS (se actualiza externamente) ===
VELAS_BUFFER = []

# === PROCESADOR PRINCIPAL DE CADA VELA NUEVA ===
def procesar_vela(vela):
    try:
        VELAS_BUFFER.append(vela)

        # Solo procesar si hay al menos 60 velas
        if len(VELAS_BUFFER) < 60:
            log(f"[⏳ ESPERA] Aún no hay 60 velas para analizar (actual: {len(VELAS_BUFFER)})")
            return

        # Construir DataFrame completo y contexto
        df = pd.DataFrame(VELAS_BUFFER[-60:])
        df["spread"] = df["high"] - df["low"]
        df["momentum"] = df["close"].diff()
        df["ema"] = df["close"].ewm(span=10).mean()
        df["rsi"] = calcular_rsi(df["close"])

        # Última vela como actual
        df_actual = pd.DataFrame([df.iloc[-1]])

        # Validar que no haya valores faltantes
        if df.isnull().values.any():
            log("[⚠️ DESCARTADA] Vela con valores nulos en indicadores.")
            return

        # Evaluar spike anticipado
        spike_detectado = evaluar_spike_anticipado(df)

        if spike_detectado:
            log("[🚫 BLOQUEO] Entrada cancelada por spike futuro.")
            return

        # Construir contexto para DRL
        contexto = construir_contexto_drl(df)

        # Ejecutar decisión DRL si todo está validado
        procesar_decision_drl(df_actual, contexto)

    except Exception as e:
        log(f"[❌ ERROR procesar_vela] {e}", "error")


# === FUNCIONES DE INDICADORES AUXILIARES ===
def calcular_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window=period).mean()
    ma_down = down.rolling(window=period).mean()
    rsi = 100 - (100 / (1 + ma_up / ma_down))
    return rsi
