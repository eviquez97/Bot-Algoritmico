# entrenar_modelo_direccion.py

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ================================
# ⚙️ Cargar y preparar el dataset
# ================================

RUTA_CSV = "data/dataset_drl.csv"
df = pd.read_csv(RUTA_CSV)

# Filtrar filas con valores nulos
df = df.dropna(subset=["direccion_futura"])

# 🧠 Variables de entrada (ajusta columnas si es necesario)
X = df[["open", "high", "low", "close", "volumen_tick"]]
y = df["direccion_futura"]

# ================================
# ⚖️ Escalado
# ================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================================
# 🧪 Entrenamiento
# ================================

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

modelo = RandomForestClassifier(n_estimators=150, random_state=42)
modelo.fit(X_train, y_train)

# ================================
# 📊 Evaluación rápida
# ================================

y_pred = modelo.predict(X_test)
print(classification_report(y_test, y_pred))

# ================================
# 💾 Guardar modelo y scaler
# ================================

joblib.dump(modelo, "modelos/modelo_direccion_rf.pkl")
joblib.dump(scaler, "modelos/scaler_direccion_rf.pkl")

print("✅ Modelo y scaler de dirección guardados correctamente.")
