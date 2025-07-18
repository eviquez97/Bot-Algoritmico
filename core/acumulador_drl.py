# core/acumulador_drl.py

import pandas as pd
import os

CSV_DRL = "data/dataset_drl.csv"

def guardar_contexto_drl(contexto, resultado, monto, mult):
    fila = {
        "score": contexto["score"],
        "ganancia_estim": contexto["ganancia_estim"],
        "porcentaje_bajistas": contexto["porcentaje_bajistas"],
        "pred_futuro": contexto["pred_futuro"],
        "resultado": resultado,  # 1 si ganó, 0 si perdió
        "monto": monto,
        "multiplicador": mult
    }

    df_fila = pd.DataFrame([fila])

    if not os.path.exists(CSV_DRL):
        df_fila.to_csv(CSV_DRL, index=False)
    else:
        df_fila.to_csv(CSV_DRL, mode='a', header=False, index=False)
