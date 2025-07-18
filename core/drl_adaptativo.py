import pandas as pd
import os
from datetime import datetime
from core.drl_estado import obtener_estado_drl
from utils.logs import log

def evaluar_rendimiento_drl():
    archivo = "data/experiencias_drl.csv"

    if not os.path.exists(archivo):
        log("[🔍 DRL] Aún no hay registros para evaluar el rendimiento.")
        return

    try:
        df = pd.read_csv(archivo)

        if "timestamp" not in df.columns or "contrato_ejecutado" not in df.columns:
            log("⚠️ [DRL] Faltan columnas necesarias en el dataset.")
            return

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
        df = df.dropna(subset=["timestamp"])

        hoy = datetime.now().date()
        df_hoy = df[df["timestamp"].dt.date == hoy]

        if df_hoy.empty:
            log("[📊 DRL] No hay entradas del DRL hoy para evaluar.")
            return

        total = len(df_hoy)
        entradas_exitosas = df_hoy[df_hoy["contrato_ejecutado"] == True]
        tasa_uso = len(entradas_exitosas) / total if total > 0 else 0

        if tasa_uso > 0.7:
            estado_drl["epsilon"] = max(0.01, estado_drl["epsilon"] - 0.01)
            log(f"[⚙️ DRL] Alto rendimiento. Epsilon reducido a {estado_drl['epsilon']:.2f}")
        elif tasa_uso < 0.3:
            estado_drl["epsilon"] = min(0.9, estado_drl["epsilon"] + 0.05)
            log(f"[⚠️ DRL] Bajo rendimiento. Epsilon aumentado a {estado_drl['epsilon']:.2f}")
        else:
            log(f"[🧪 DRL] Rendimiento estable. Epsilon mantiene en {estado_drl['epsilon']:.2f}")

    except Exception as e:
        log(f"[❌ ERROR DRL RENDIMIENTO] {e}")

