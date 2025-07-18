import pandas as pd
from core.contexto import construir_contexto_para_spike
from core.ia_spike import evaluar_spike_ia

RUTA_CSV = "data/contexto_historico.csv"
COLUMNAS_REQUERIDAS = ["open", "high", "low", "close"]

print("🔍 Cargando CSV...")
try:
    df_csv = pd.read_csv(RUTA_CSV)

    if df_csv.shape[0] < 130:
        print(f"❌ CSV insuficiente: Solo {df_csv.shape[0]} filas")
        exit()

    # Excluimos la última vela por seguridad
    df_filtrado = df_csv.tail(150).iloc[:-1]

    columnas_faltantes = [col for col in COLUMNAS_REQUERIDAS if col not in df_filtrado.columns]
    if columnas_faltantes:
        print(f"❌ Faltan columnas requeridas: {columnas_faltantes}")
        exit()

    if df_filtrado[COLUMNAS_REQUERIDAS].isnull().any().any():
        print("❌ Columnas clave contienen NaNs. No se puede construir contexto.")
        exit()

    print("🔧 Construyendo contexto Spike IA (últimas 119 velas válidas)...")
    df_contexto = construir_contexto_para_spike(df_filtrado)

    if df_contexto is None or df_contexto.shape[0] < 30:
        print("❌ Contexto inválido o insuficiente.")
        exit()

    print("✅ Contexto construido correctamente.")
    print("📊 Últimas 3 filas del contexto:")
    print(df_contexto.tail(3))

    print("🧠 Ejecutando predicción Spike IA...")
    pred = evaluar_spike_ia(df_contexto)

    if not pred or not isinstance(pred, dict):
        print("❌ No se obtuvo predicción válida.")
        exit()

    rf = pred.get("rf_spike", 0)
    lstm = pred.get("lstm_spike", 0)
    visual = pred.get("visual_spike", 0)

    print(f"🧠 SPIKE IA V5 - RF: {rf:.2f} | LSTM: {lstm:.2f} | Visual: {visual:.2f}")

    votos = sum([rf >= 0.25, lstm >= 0.15, visual >= 0.15])
    if votos >= 2 or any([rf >= 0.60, lstm >= 0.60, visual >= 0.60]):
        print("🛡️ Spike anticipado detectado. Se debería cerrar el contrato.")
    else:
        print("🟢 No se detecta spike inminente. Contrato puede mantenerse abierto.")

except Exception as e:
    print(f"❌ ERROR en test: {e}")
