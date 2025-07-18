# core/contexto.py

import pandas as pd
import numpy as np
from utils.logs import log

COLUMNAS_SPIKE_RF = [
    'fuerza_cuerpo', 'fuerza_mecha', 'bajistas', 'rsi',
    'momentum', 'spread', 'score', 'ema', 'variacion'
]

COLUMNAS_DRL = [
    'score', 'rsi', 'momentum', 'spread',
    'ema', 'variacion', 'fuerza_cuerpo', 'fuerza_mecha',
    'mecha_superior', 'mecha_inferior', 'bajistas', 'cuerpo',
    'Q0', 'Q1', 'Q2', 'Q3', 'ema_diff'
]

def calcular_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean().replace(0, 1e-10)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.replace([np.inf, -np.inf], 50).fillna(50)

def construir_contexto(df, cantidad=60):
    try:
        if df is None or df.empty:
            log("[❌ CONTEXTO] DataFrame vacío o None recibido.")
            return None

        columnas_requeridas = ["open", "high", "low", "close"]
        for col in columnas_requeridas:
            if col not in df.columns:
                log(f"[❌ CONTEXTO] Falta columna requerida: {col}")
                return None

        df = df.copy()
        df = df.tail(120).reset_index(drop=True)

        try:
            df["cuerpo"] = abs(df["close"] - df["open"])
            df["variacion"] = (df["close"] - df["open"]) / df["open"]
            df["fuerza_cuerpo"] = df["cuerpo"] / (df["high"] - df["low"]).replace(0, 1)
            df["mecha_superior"] = df["high"] - df[["close", "open"]].max(axis=1)
            df["mecha_inferior"] = df[["close", "open"]].min(axis=1) - df["low"]
            df["fuerza_mecha"] = df["mecha_superior"] + df["mecha_inferior"]
            df["spread"] = df["high"] - df["low"]
            df["score"] = df["variacion"].rolling(window=5, min_periods=1).mean()
            df["rsi"] = calcular_rsi(df["close"])
            df["ema"] = df["close"].ewm(span=10, adjust=False).mean()
            df["momentum"] = df["close"].diff()
            df["bajistas"] = (df["close"] < df["open"]).astype(int)
            df["ema_diff"] = df["ema"].diff()
            df["Q0"] = df["close"].rolling(10).quantile(0.0)
            df["Q1"] = df["close"].rolling(10).quantile(0.25)
            df["Q2"] = df["close"].rolling(10).quantile(0.5)
            df["Q3"] = df["close"].rolling(10).quantile(0.75)
        except Exception as e:
            log(f"[⚠️ CONTEXTO] Error en cálculo de columnas: {e}")

        df = df.replace([np.inf, -np.inf], 0.0).bfill().ffill()

        if len(df) < cantidad:
            log(f"[⚠️ CONTEXTO] Solo {len(df)} filas válidas tras limpieza. Requiere mínimo {cantidad}, pero se continúa.")

        df = df.tail(cantidad).reset_index(drop=True)

        columnas_faltantes = [col for col in COLUMNAS_DRL if col not in df.columns]
        if columnas_faltantes:
            log(f"[❌ CONTEXTO] Faltan columnas requeridas para DRL: {columnas_faltantes}")
            return None

        df_final = df[COLUMNAS_DRL].copy()
        return df_final

    except Exception as e:
        log(f"[❌ ERROR CONTEXTO FATAL] {e}")
        return None

def construir_contexto_para_spike(df, cantidad=60):
    try:
        if df is None or df.empty:
            log("[❌ CONTEXTO SPIKE] DataFrame vacío.")
            return None

        columnas_requeridas = ["open", "high", "low", "close"]
        for col in columnas_requeridas:
            if col not in df.columns:
                log(f"[❌ CONTEXTO SPIKE] Falta columna: {col}")
                return None

        df = df.copy()
        df = df.tail(121).reset_index(drop=True)

        if pd.isna(df.iloc[-1]["close"]):
            df = df.iloc[:-1]

        try:
            df["cuerpo"] = abs(df["close"] - df["open"])
            df["variacion"] = (df["close"] - df["open"]) / df["open"]
            df["fuerza_cuerpo"] = df["cuerpo"] / (df["high"] - df["low"]).replace(0, 1)
            df["fuerza_mecha"] = ((df["high"] - df["low"]) - df["cuerpo"]) / (df["high"] - df["low"]).replace(0, 1)
            df["spread"] = df["high"] - df["low"]
            df["ema"] = df["close"].ewm(span=10, adjust=False).mean()
            df["rsi"] = calcular_rsi(df["close"])
            df["momentum"] = df["close"].diff()
            df["score"] = df["variacion"].rolling(window=5, min_periods=1).mean()
            df["bajistas"] = (df["close"] < df["open"]).astype(int)
            df["alcista"] = (df["close"] > df["open"]).astype(int)  # ✅ añadida esta línea
        except Exception as e:
            log(f"[⚠️ CONTEXTO SPIKE] Error en cálculo de columnas: {e}")

        columnas_finales = COLUMNAS_SPIKE_RF + ["open", "close", "high", "low", "alcista"]  # ✅ añadidos 'low' y 'alcista'

        df_nan_check = df[columnas_finales] if all(col in df.columns for col in columnas_finales) else df.dropna()
        nan_por_columna = df_nan_check.isna().sum()
        total_nans = nan_por_columna.sum()
        if total_nans > 0:
            log(f"[🧪 SPIKE] NaNs encontrados: {dict(nan_por_columna[nan_por_columna > 0])}")

        df_spike = df_nan_check.replace([np.inf, -np.inf], 0.0).dropna().bfill().ffill()

        if len(df_spike) < 30:
            log(f"[❌ CONTEXTO SPIKE] Insuficiente para spike IA. Requiere 30+, tiene {len(df_spike)}")
            return None

        df_spike = df_spike.tail(cantidad).reset_index(drop=True)
        log(f"[🧠 CONTEXTO SPIKE] Filas válidas tras limpieza: {len(df_spike)}")

        return df_spike

    except Exception as e:
        log(f"[❌ ERROR CONTEXTO SPIKE] {e}")
        return None

