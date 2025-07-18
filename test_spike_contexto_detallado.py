import pandas as pd
from core.contexto import construir_contexto_para_spike

CSV_PATH = "data/contexto_historico.csv"

def analizar_columnas(df):
    print("\n🔎 ANÁLISIS COLUMNA POR COLUMNA:")
    for col in df.columns:
        n_nan = df[col].isna().sum()
        dtype = df[col].dtype
        muestra = df[col].dropna().head(3).tolist()
        print(f" - {col}: NaNs={n_nan} | Tipo={dtype} | Ejemplo={muestra}")

def test_contexto_spike_detallado():
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"[🔬 TEST SPIKE IA] DF original recibido: {len(df)} filas")

        df_reciente = df.tail(60)
        print(f"[🔬 TEST SPIKE IA] Últimas 60 filas cargadas")

        # Análisis previo a construir contexto
        analizar_columnas(df_reciente)

        # Ejecutamos la función real del bot
        contexto = construir_contexto_para_spike(df_reciente)

        if contexto is None:
            print("[❌ CONTEXTO SPIKE] Error: función devolvió None")
            return

        print(f"[✅ CONTEXTO SPIKE] Filas tras limpieza: {len(contexto)}")
        analizar_columnas(contexto)

        if len(contexto) < 30:
            print(f"[❌ ERROR] Aún tras limpieza profunda, solo hay {len(contexto)} filas. Algo las está eliminando.")
        else:
            print("[✅ CONTEXTO VÁLIDO] Spike IA puede ejecutarse sin bloqueo.")

    except Exception as e:
        print(f"[❌ ERROR TEST SPIKE DETALLADO] {e}")

if __name__ == "__main__":
    test_contexto_spike_detallado()
