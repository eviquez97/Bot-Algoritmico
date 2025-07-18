import pandas as pd
from core.contexto import construir_contexto_para_spike

CSV = "data/contexto_historico.csv"

df_csv = pd.read_csv(CSV).tail(120)
print(f"\n[🔍 TEST CONTEXTO SPIKE REAL] Últimas 120 filas cargadas: {len(df_csv)}")

df_contexto = construir_contexto_para_spike(df_csv)

if df_contexto is None:
    print("\n[❌ CONTEXTO] La función retornó None")
else:
    print(f"\n[✅ CONTEXTO] Filas retornadas: {len(df_contexto)}")
    print("\n[📊 COLUMNA A COLUMNA]")
    for col in df_contexto.columns:
        print(f" - {col}: NaNs={df_contexto[col].isna().sum()} | Ejemplo: {df_contexto[col].values[:3]}")
