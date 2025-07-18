import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import os

# === RUTAS ===
CSV_PATH = "data/dataset_spike_monstruo_limpio.csv"
MODELO_PATH = "modelos/model_spike.pkl"
SCALER_PATH = "modelos/scaler_rf_2.pkl"

# === COLUMNAS USADAS ===
COLUMNAS_USADAS = [
    'fuerza_cuerpo', 'fuerza_mecha', 'bajistas', 'rsi',
    'momentum', 'spread', 'score'
]

# === CARGA Y LIMPIEZA ===
try:
    df = pd.read_csv(CSV_PATH)
except Exception as e:
    print(f"[❌ ERROR] No se pudo cargar el archivo: {e}")
    exit()

df = df.dropna(subset=COLUMNAS_USADAS + ['spike'])

# === BALANCEO ===
df_positivos = df[df["spike"] == 1]
df_negativos = df[df["spike"] == 0]

min_len = min(len(df_positivos), len(df_negativos))
df_balanceado = pd.concat([
    df_positivos.sample(min_len, random_state=42),
    df_negativos.sample(min_len, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

# === SEPARACIÓN ===
X = df_balanceado[COLUMNAS_USADAS]
y = df_balanceado["spike"]

# === ESCALADO ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === SPLIT Y ENTRENAMIENTO ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === EVALUACIÓN ===
y_pred = model.predict(X_test)
print("\n===== [📊 EVALUACIÓN RF BALANCEADO SPIKE] =====")
print(classification_report(y_test, y_pred))

# === GUARDADO ===
os.makedirs("modelos", exist_ok=True)
joblib.dump(model, MODELO_PATH)
joblib.dump(scaler, SCALER_PATH)

print(f"\n[✅ MODELO GUARDADO] {MODELO_PATH}")
print(f"[✅ SCALER GUARDADO] {SCALER_PATH}")
