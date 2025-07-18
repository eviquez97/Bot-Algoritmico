import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Cargar dataset limpio
RUTA_CSV = "data/dataset_spike_monstruo_limpio.csv"
df = pd.read_csv(RUTA_CSV)

# Filtrado básico
df = df.dropna()
columnas = ['fuerza_cuerpo', 'fuerza_mecha', 'bajistas', 'rsi', 'momentum', 'spread', 'score', 'spike']
df = df[columnas]

# Separar variables
X = df.drop(columns=["spike"])
y = df["spike"]

# Balancear dataset
n_spike = sum(y == 1)
n_no_spike = sum(y == 0)
minimo = min(n_spike, n_no_spike)

df_spike = df[df["spike"] == 1].sample(n=minimo, random_state=42)
df_no_spike = df[df["spike"] == 0].sample(n=minimo, random_state=42)
df_balanceado = pd.concat([df_spike, df_no_spike]).sample(frac=1, random_state=42)

X_bal = df_balanceado.drop(columns=["spike"])
y_bal = df_balanceado["spike"]

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_bal)

# Entrenar modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_bal, test_size=0.2, random_state=42)
modelo.fit(X_train, y_train)

# Evaluar
y_pred = modelo.predict(X_test)
print("\n===== [📊 EVALUACIÓN RF BALANCEADO SPIKE] =====")
print(classification_report(y_test, y_pred))

# Guardar modelo y scaler
with open("modelos/model_spike.pkl", "wb") as f:
    pickle.dump(modelo, f)
with open("modelos/scaler_rf_2.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n[✅ MODELO GUARDADO] modelos/model_spike.pkl")
print("[✅ SCALER GUARDADO] modelos/scaler_rf_2.pkl")
