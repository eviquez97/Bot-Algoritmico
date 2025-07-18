# core/drl_estado.py

import numpy as np

# === GENERADOR DE ESTADO PARA DRL ===
def obtener_estado_drl(df_actual):
    try:
        score = df_actual["score"].iloc[0]
        bajistas = df_actual["porcentaje_bajistas"].iloc[0]
        ganancia = df_actual["ganancia_estimada"].iloc[0]
        pred_futuro = df_actual["pred_futuro"].iloc[0]
        spread = df_actual["spread"].iloc[0]
        momentum = df_actual["momentum"].iloc[0]
        rsi = df_actual["rsi"].iloc[0]
        ema = df_actual["ema"].iloc[0]

        estado = np.array([score, bajistas, ganancia, pred_futuro, spread, momentum, rsi, ema])
        return estado.reshape(1, -1)

    except Exception as e:
        print(f"[❌ ERROR ESTADO DRL] {type(e).__name__} - {e}")
        return np.zeros((1, 8))
