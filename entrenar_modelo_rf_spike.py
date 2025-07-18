# entrenar_modelo_rf_spike.py

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Cargar dataset
df = pd.read_csv("data/dataset_operativo.csv")

# Columnas actuales confirmadas
COLUMNAS_ENTRADA = [
    'open', 'high', 'low', 'close', 'spread', 'fuerza_cuerpo',
    'fuerza_mecha', 'score', 'rsi', 'ema', 'momentum', 'variacion', 'alcista'
]

# Validación
for col in COLUMNAS_ENTRADA:
    if col not in df.columns:
        print(f"❌ FALTA LA COLUMNA: {col}")
        exit()

X = df[COLUMNAS_ENTRADA]
y = df["spike_real"]

# Preprocesamiento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entrenar modelo RF
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Guardar modelo y scaler
joblib.dump(rf, "modelos/model_spike.pkl")
joblib.dump(scaler, "modelos/scaler_rf_2.pkl")
print("✅ Modelo RF Spike y scaler guardados.")
