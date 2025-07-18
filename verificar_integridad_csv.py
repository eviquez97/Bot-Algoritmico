# verificar_integridad_csv.py

import pandas as pd
from utils.logs import log

RUTA = "data/contexto_historico.csv"
COLUMNAS_ESPERADAS = [
    'epoch', 'open', 'high', 'low', 'close', 'spread',
    'momentum', 'variacion', 'score', 'rsi', 'ema',
    'fuerza_cuerpo', 'mecha_superior', 'mecha_inferior',
    'fuerza_mecha', 'bajistas'
]

try:
    df = pd.read_csv(RUTA)

    log(f"[📂 CSV DETECTADO] {RUTA} con {len(df)} filas y {len(df.columns)} columnas.")

    columnas_invalidas = [col for col in df.columns if col not in COLUMNAS_ESPERADAS]
    columnas_faltantes = [col for col in COLUMNAS_ESPERADAS if col not in df.columns]

    if columnas_invalidas:
        log(f"[❌ COLUMNAS EXTRAÑAS] No esperadas: {columnas_invalidas}")
    if columnas_faltantes:
        log(f"[❌ COLUMNAS FALTANTES] Se requieren: {columnas_faltantes}")

    columnas_correctas = (df.columns.tolist() == COLUMNAS_ESPERADAS)
    if columnas_correctas:
        log("[✅ ESTRUCTURA] Columnas en orden y completas.")
    else:
        log("[⚠️ ESTRUCTURA] Columnas desordenadas o incorrectas.")

    filas_nan = df[df.isna().any(axis=1)]
    if not filas_nan.empty:
        log(f"[⚠️ NaN DETECTADOS] Filas con valores faltantes: {len(filas_nan)}")

    filas_validas = df.dropna()
    log(f"[📊 VALIDACIÓN] Filas totales: {len(df)} | Válidas (sin NaN): {len(filas_validas)}")

    if len(filas_validas) >= 60:
        log(f"[✅ LISTO] Hay suficientes velas válidas ({len(filas_validas)}) para Spike IA.")
    else:
        log(f"[❌ INSUFICIENTE] Solo hay {len(filas_validas)} velas válidas para Spike IA (mínimo 60).")

except Exception as e:
    log(f"[❌ ERROR CSV] {e}")
