import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from core.contexto import construir_contexto
from sklearn.preprocessing import StandardScaler
from utils.logs import log

# Paths
CSV_PATH = "data/contexto_historico.csv"
MODELO_DRL_PATH = "modelos/modelo_drl.keras"
MODELO_GANANCIA_PATH = "modelos/modelo_ganancia_rf.pkl"
SCALER_GANANCIA_PATH = "modelos/scaler_ganancia_rf.pkl"

# Columnas esperadas por DRL (orden exacto)
COLUMNAS_DRL = [
    'score', 'rsi', 'momentum', 'spread',
    'ema', 'variacion', 'fuerza_cuerpo', 'fuerza_mecha',
    'mecha_superior', 'mecha_inferior', 'bajistas', 'cuerpo',
    'Q0', 'Q1', 'Q2', 'Q3', 'ema_diff'
]

COLUMNAS_GANANCIA = ['score', 'rsi', 'momentum', 'spread']

try:
    df = pd.read_csv(CSV_PATH)
    if df.empty:
        raise Exception("CSV vacío")

    df_contexto = construir_contexto(df, cantidad=60)
    if df_contexto is None or df_contexto.shape != (60, 17):
        raise Exception(f"Contexto inválido. Shape actual: {df_contexto.shape if df_contexto is not None else 'None'}")

    # Reescalar X para modelo DRL
    X_drl = df_contexto[COLUMNAS_DRL].values
    X_drl = np.array(X_drl).reshape(1, 60, 17)

    # Cargar modelos
    modelo_drl = load_model(MODELO_DRL_PATH)
    modelo_ganancia = joblib.load(MODELO_GANANCIA_PATH)
    scaler_ganancia = joblib.load(SCALER_GANANCIA_PATH)

    log("✅ Modelo Spike Visual (SCS-VISION X) cargado exitosamente.")
    log("✅ Modelo RF de Ganancia cargado correctamente.")
    log("✅ Modelo RF de Dirección cargado correctamente.")

    print(f"📄 CSV cargado con {len(df)} filas.")
    print(f"🔍 Evaluando decisión DRL...\n")

    # Predicción DRL
    score = modelo_drl.predict(X_drl)[0][0]

    # Estimación ganancia
    ultima_fila = df_contexto.tail(1)[COLUMNAS_GANANCIA]
    X_ganancia = scaler_ganancia.transform(ultima_fila)
    ganancia_estim = modelo_ganancia.predict(X_ganancia)[0]

    # Monto y multiplicador
    capital = 500.0
    riesgo = min(max(score, 0.01), 1.0)
    monto = round(capital * riesgo * 0.1, 2)
    multiplicador = 100.0 if riesgo < 0.5 else 300.0
    duracion = 60 if ganancia_estim < 1 else 180

    print(f"📊 RESULTADO DECISIÓN DRL:")
    print(f"📌 Acción: 0")
    print(f"📈 Score DRL: {score:.4f}")
    print(f"💰 Monto sugerido: ${monto}")
    print(f"🎯 Multiplicador: {multiplicador}")
    print(f"💡 Ganancia estimada: ${ganancia_estim:.2f}")
    print(f"⏳ Duración estimada: {duracion} segundos")
    print(f"🚦 Entrada permitida: {'Sí' if monto > 0 else 'No'}")

except Exception as e:
    print(f"[❌ ERROR TEST DRL] {e}")
