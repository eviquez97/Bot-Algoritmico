# test_origen_spike_llamada.py
import traceback
import pandas as pd
from core.ia_spike import evaluar_spike_ia

def test_llamada_spike(df):
    print("\n📍 LLAMADA SPIKE IA INTERCEPTADA")
    print(f"[🔢 FILAS RECIBIDAS] {len(df)}")
    print(f"[🧪 COLUMNAS] {df.columns.tolist()}")
    print(f"[🔍 EJEMPLO HEAD]\n{df.head(3)}")

    # Traza de quién llama esta función
    print("\n🧬 PILA DE LLAMADAS:")
    traceback.print_stack()

    # Ejecutar evaluación real
    resultado = evaluar_spike_ia(df)
    print("\n[✅ RESULTADO SPIKE IA]", resultado)

# Simulación: tú mismo debes colocar esto temporalmente en main.py o donde se llame evaluar_spike_ia
# Reemplaza:
#     evaluar_spike_ia(df_csv.tail(60))
# Por:
#     from test_origen_spike_llamada import test_llamada_spike
#     test_llamada_spike(df_csv.tail(60))
