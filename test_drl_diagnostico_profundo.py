# test_drl_diagnostico_profundo.py

import pandas as pd
import numpy as np
import joblib
from core.ia_drl import procesar_decision_drl
from core.contexto import construir_contexto
from core.estado import contrato_activo
from utils.logs import log

CSV = "data/contexto_historico.csv"

def test_drl_con_contexto():
    if contrato_activo:
        print("[⛔ TEST DRL] contrato_activo = True. No se permite operar.")
        return

    try:
        df_csv = pd.read_csv(CSV)
        print(f"[📥 CSV] Archivo cargado con {len(df_csv)} velas.")

        if len(df_csv) < 60:
            print("[⛔ TEST DRL] No hay suficientes filas para construir contexto.")
            return

        df = df_csv.tail(120).copy()

        print(f"[🧪 TEST] Columnas disponibles en df: {list(df.columns)}")

        contexto = construir_contexto(df, cantidad=60)
        if contexto is None:
            print("[❌ TEST DRL] Error al construir contexto.")
            return

        print(f"[✅ CONTEXTO] Contexto construido con columnas: {list(contexto.keys())}")

        columnas_drl = joblib.load("modelos/columnas_drl.pkl")
        faltantes = [col for col in columnas_drl if col not in contexto]
        if faltantes:
            print(f"[❌ TEST DRL] Columnas faltantes para DRL: {faltantes}")
        else:
            print("[✅ TEST DRL] Todas las columnas requeridas están presentes.")

        capital = 500.0
        multiplicadores = [100, 200, 300, 400]
        decision = procesar_decision_drl(contexto, capital, multiplicadores)

        print("\n========= RESULTADO DRL =========")
        print(f"Permitir entrada : {decision['permitir_entrada']}")
        print(f"Score DRL        : {decision['score']:.4f}")
        print(f"Acción elegida   : {decision['accion']}")
        print(f"Monto            : ${decision['monto']:.2f}")
        print(f"Multiplicador    : {decision['multiplicador']}")
        print(f"Ganancia esperada: ${decision['ganancia_esperada']:.2f}")
        print(f"Duración estimada: {decision['duracion_estimada']}s")
        print("=================================\n")

    except Exception as e:
        print(f"[❌ TEST DRL ERROR] {e}")

if __name__ == "__main__":
    test_drl_con_contexto()
