import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# === Cargar dataset
df = pd.read_csv("data/dataset_operativo.csv")

# === Cargar columnas correctas para DRL (blindado)
COLUMNAS_DRL = joblib.load("modelos/columnas_drl.pkl")

# === Modelo Spike IA
X_spike = df[["open", "high", "low", "close", "rsi", "ema", "momentum", "score", "spread"]]
y_spike = df["spike_real"]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_spike, y_spike, test_size=0.2, random_state=42)
scaler_spike = StandardScaler()
X_train_s = scaler_spike.fit_transform(X_train_s)
X_test_s = scaler_spike.transform(X_test_s)

modelo_spike = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_spike.fit(X_train_s, y_train_s)

print("🔎 Evaluación modelo Spike IA:")
print(classification_report(y_test_s, modelo_spike.predict(X_test_s)))
joblib.dump(modelo_spike, "modelos/model_spike.pkl")
joblib.dump(scaler_spike, "modelos/scaler_spike.pkl")

# === Modelo DRL
X_drl = df[COLUMNAS_DRL]
y_drl = df["operacion_exitosa"]

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_drl, y_drl, test_size=0.2, random_state=42)
scaler_drl = StandardScaler()
X_train_d = scaler_drl.fit_transform(X_train_d)
X_test_d = scaler_drl.transform(X_test_d)

modelo_drl = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_drl.fit(X_train_d, y_train_d)

print("🔎 Evaluación modelo DRL:")
print(classification_report(y_test_d, modelo_drl.predict(X_test_d)))
joblib.dump(modelo_drl, "modelos/modelo_drl.pkl")
joblib.dump(scaler_drl, "modelos/scaler_drl.pkl")

print("✅ Modelos entrenados y guardados.")

