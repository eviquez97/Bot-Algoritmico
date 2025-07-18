# test_spike_rf_ultimas_100.py

import pandas as pd
import joblib

# Cargar modelo y scaler
modelo_rf = joblib.load("modelos/model_spike.pkl")
scaler = joblib.load("modelos/scaler_rf_2.pkl")

# Columnas usadas en el entrenamiento
columnas = list(scaler.feature_names_in_)

# Cargar dataset
df = pd.read_csv("data/dataset_spike_monstruo_limpio.csv")

# Filtrar columnas necesarias y eliminar nulos
df_filtrado = df[columnas].dropna().astype('float64')

# Seleccionar últimas 100 filas
df_test = df_filtrado.tail(100)

# Aplicar scaler
X_scaled = scaler.transform(df_test)

# Predecir probabilidades
probas = modelo_rf.predict_proba(X_scaled)

# Extraer probabilidades de clase 1 (spike real)
predicciones = [round(prob[1], 2) for prob in probas]

# Mostrar resultados
print("\n🔬 Predicciones SPIKE RF (últimas 100 filas):\n")
print(predicciones)
