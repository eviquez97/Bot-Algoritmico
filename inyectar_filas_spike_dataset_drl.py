import pandas as pd
from datetime import datetime

CSV_PATH = "data/dataset_drl.csv"

# Crear 2 filas falsas con spike leve para pruebas reales
nuevas_filas = [
    {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "score": 18.5,
        "ganancia_estimada": 42.0,
        "porcentaje_bajistas": 87.0,
        "prediccion_futura": 0.77,
        "ema": 18300.12,
        "rsi": 14.8,
        "momentum": -1.43,
        "spread": 1.12,
        "rf_spike": 0.40,
        "lstm_spike": 0.30,
        "visual_spike": 0.35,
        "ultima_direccion": 0.0,
        "accion": 1,
        "monto": 5.0,
        "multiplicador": 200,
        "exito": 1,
        "Q0": 0.0,
        "Q1": 0.0,
        "Q2": 0.0,
        "Q3": 0.0,
        "ema_diff": -0.93
    },
    {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "score": 21.1,
        "ganancia_estimada": 58.3,
        "porcentaje_bajistas": 90.0,
        "prediccion_futura": 0.82,
        "ema": 18210.78,
        "rsi": 17.2,
        "momentum": -1.62,
        "spread": 1.30,
        "rf_spike": 0.52,
        "lstm_spike": 0.38,
        "visual_spike": 0.41,
        "ultima_direccion": 0.0,
        "accion": 2,
        "monto": 6.0,
        "multiplicador": 300,
        "exito": 1,
        "Q0": 0.0,
        "Q1": 0.0,
        "Q2": 0.0,
        "Q3": 0.0,
        "ema_diff": -1.22
    }
]

# Cargar dataset y agregar nuevas filas respetando columnas existentes
df = pd.read_csv(CSV_PATH)
df_nuevas = pd.DataFrame(nuevas_filas)

# Filtrar columnas para evitar errores
columnas_validas = [col for col in df.columns if col in df_nuevas.columns]
df_nuevas = df_nuevas[columnas_validas]

# Añadir y sobrescribir
df_actualizado = pd.concat([df, df_nuevas], ignore_index=True)
df_actualizado.to_csv(CSV_PATH, index=False)
print(f"✅ Inyectadas {len(df_nuevas)} filas con spike leve en {CSV_PATH}")
