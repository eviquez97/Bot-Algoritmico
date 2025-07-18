# test_spike_interceptor.py

import pandas as pd
import traceback
from core.ia_spike import evaluar_spike_ia as spike_original
from utils.logs import log

def evaluar_spike_ia_interceptada(df):
    print("\n📍 [🔬 INTERCEPTOR ACTIVADO] Se llamó a evaluar_spike_ia(df)")
    print(f"[🔢 FILAS RECIBIDAS] {len(df)}")
    
    if isinstance(df, pd.DataFrame):
        print(f"[📊 COLUMNAS RECIBIDAS] {list(df.columns)}")
        print("[📉 NULOS POR COLUMNA]")
        print(df.isna().sum())
    else:
        print("[❌ ERROR] df no es un DataFrame válido")

    if len(df) < 60:
        print(f"[⚠️ ALERTA] Se recibieron {len(df)} filas, pero se esperaban al menos 60 para Spike IA.")

    print("\n🧬 PILA DE LLAMADAS:")
    traceback.print_stack()

    print("\n🧠 [RE-ENVIANDO A SPIKE ORIGINAL]")
    return spike_original(df)

# Sobreescribe la función original (esto debe importarse antes de que spike_ia se use)
import core.ia_spike
core.ia_spike.evaluar_spike_ia = evaluar_spike_ia_interceptada
