import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import os
from core.contexto import construir_contexto

# === CONFIGURACIÓN ===
CSV_PATH = "data/contexto_historico.csv"
MODELO_PATH = "modelos/modelo_prediccion_futura.pkl"
SCALER_PATH = "modelos/escalador_futuro.pkl"

# === CARGA Y PREPARACIÓN ===
df = pd.read_csv(CSV_PATH)
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Verificar existencia de columna objetivo
if 'futuro' not in df.columns:
    raise ValueError("❌ La columna 'futuro' no está presente en el CSV.")

# === Construcción de contexto ===
X = construir_contexto(df, cantidad=30)
y = df["futuro"].iloc[30:].reset_index(drop=True)

# Conversión segura a DataFrame (por si X es lista de dicts)
if isinstance(X, list):
    X = pd.DataFrame(X)

# Validación
if not isinstance(X, pd.DataFrame):
    raise ValueError("❌ Error: el contexto no se generó como DataFrame válido.")

# Limpieza final
X = X.dropna()
y = y.loc[X.index]  # sincronizar con X limpio

# === ESCALADO ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === ENTRENAMIENTO ===
modelo = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
modelo.fit(X_scaled, y)

# === GUARDADO ===
os.makedirs("modelos", exist_ok=True)
joblib.dump(modelo, MODELO_PATH)
joblib.dump(scaler, SCALER_PATH)

print("✅ Modelo de predicción futura entrenado y guardado correctamente.")
