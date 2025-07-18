import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

# Ruta del CSV generado automáticamente
RUTA_CSV = "data/contexto_historico.csv"
RUTA_MODELO = "modelos/modelo_ganancia_rf.pkl"

# Columnas que realmente se usan en tiempo real (las del contexto real)
COLUMNAS_REALES = [
    "score", "futuro", "bajistas", "visual_spike", "rf_spike",
    "lstm_spike", "ema_diff", "rsi", "momentum", "spread",
    "monto", "multiplicador", "ganancia_futura"
]

# Verificamos que el archivo exista
if not os.path.exists(RUTA_CSV):
    raise FileNotFoundError(f"No se encuentra el archivo: {RUTA_CSV}")

# Cargamos y verificamos las columnas
df = pd.read_csv(RUTA_CSV)

for col in COLUMNAS_REALES:
    if col not in df.columns:
        raise Exception(f"Falta la columna requerida: {col}")

df = df[COLUMNAS_REALES].dropna()

# Separación de variables
X = df.drop(columns=["ganancia_futura"])
y = df["ganancia_futura"]

# División en entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento del modelo
modelo = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
modelo.fit(X_train, y_train)

# Evaluación
y_pred = modelo.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Guardado del modelo
joblib.dump(modelo, RUTA_MODELO)

print(f"✅ Modelo entrenado. MSE en test: {mse:.4f}")
print(f"💾 Modelo guardado como {RUTA_MODELO}")
