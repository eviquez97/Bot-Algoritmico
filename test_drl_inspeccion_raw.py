# test_drl_inspeccion_raw.py

import pandas as pd
import numpy as np
import joblib
from core.contexto import construir_contexto
from core.estado import contrato_activo
from core.ia_modelos import modelo_drl

CSV = "data/contexto_historico.csv"

def test_modelo_drl_raw():
    if contrato_activo:
        print("[⛔ TEST] contrato_activo = True. No se permite operar.")
        return

    try:
        df_csv = pd.read_csv(CSV)
        df = df_csv.tail(120).copy()

        contexto = construir_contexto(df, cantidad=60)
        if contexto is None:
            print("[❌ ERROR] Contexto no construido.")
            return

        columnas_drl = joblib.load("modelos/columnas_drl.pkl")
        X_drl = pd.DataFrame([contexto])[columnas_drl]

        print(f"[🧪 INPUT AL MODELO DRL] {X_drl}")

        y_pred = modelo_drl.predict(X_drl)
        print(f"\n[🔍 OUTPUT DEL MODELO DRL] Predicción cruda:\n{y_pred}")

        score_drl = float(y_pred[0][0])
        accion = int(np.argmax(y_pred[0]))

        print(f"\n[📊 RESULTADO INTERNO]")
        print(f"Score DRL: {score_drl:.4f}")
        print(f"Acción elegida (argmax): {accion}")

    except Exception as e:
        print(f"[❌ ERROR TEST RAW] {e}")

if __name__ == "__main__":
    test_modelo_drl_raw()
