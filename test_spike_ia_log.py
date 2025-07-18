# test_spike_ia_log.py

import pandas as pd
import numpy as np
from core.ia_spike import evaluar_spike_ia
from modelos.modelos_spike import scaler_rf_2

# 🧪 Simulamos un DataFrame con 60 filas y las columnas necesarias
columnas = list(scaler_rf_2.feature_names_in_)
data = []

for _ in range(60):
    fila = {
        'open': 100.0,
        'close': 99.5,  # Vela bajista
        'high': 100.2,
        'low': 99.4,
        'spread': np.random.rand(),
        'momentum': np.random.rand(),
        'score': np.random.rand(),
        'ema': np.random.rand(),
        'variacion': np.random.rand(),
        'fuerza_cuerpo': np.random.rand(),
        'fuerza_mecha': np.random.rand(),
        'bajistas': np.random.randint(0, 5),
        'rsi': np.random.uniform(30, 70),
    }
    data.append(fila)

df_fake = pd.DataFrame(data)

# 🔎 Ejecutamos la función y observamos si imprime el log correcto
print("🔍 Ejecutando test sobre evaluar_spike_ia(df_fake)...\n")
resultado = evaluar_spike_ia(df_fake)

# 📋 Imprimimos resultado bruto del diccionario
print("\n📊 Resultado retornado:")
print(resultado)
