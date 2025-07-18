# test_drl_diagnostico_avanzado.py

import pandas as pd
from core.ia_drl import procesar_decision_drl
from utils.logs import log

print("🧪 TEST DRL AVANZADO | Diagnóstico profundo de entrada")

# Cargar el dataset exacto con el que se entrena DRL
CSV_PATH = "data/dataset_drl.csv"

try:
    df = pd.read_csv(CSV_PATH)
    print(f"✅ Dataset DRL cargado: {len(df)} filas")

    if len(df) < 60:
        print("❌ ERROR: Menos de 60 filas. No se puede formar contexto.")
        exit()

    columnas = df.columns.tolist()
    print(f"📋 Columnas detectadas: {columnas}")

    contexto = df.tail(60)
    print(f"🔍 Última fila contexto:\n{contexto.iloc[-1].to_dict()}")

    capital_simulado = 300.0
    multiplicadores_simulados = [100, 200, 300, 400]

    print("\n🚀 Ejecutando DRL con contexto real...")
    decision = procesar_decision_drl(contexto, capital_simulado, multiplicadores_simulados)

    print("\n📊 RESULTADO DECISIÓN DRL:")
    for clave, valor in decision.items():
        print(f"🔸 {clave}: {valor}")

    if not decision["permitir_entrada"]:
        print("\n⚠️ DRL NO PERMITE ENTRADA. RAZONES POSIBLES:")
        print("- Modelo retornó score bajo o acción inválida")
        print("- Ganancia estimada <= 0")
        print("- Monto o multiplicador = 0")
        print("- Fallo en reshape o columnas faltantes")

except Exception as e:
    print(f"❌ EXCEPCIÓN CRÍTICA DURANTE TEST: {e}")
