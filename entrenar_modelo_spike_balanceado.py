# entrenar_modelo_spike_balanceado.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import classification_report
import joblib
import os

# Cargar CSV limpio
df = pd.read_csv("data/dataset_spike_monstruo_limpio.csv")

# Verificamos que exista la columna 'spike' como etiqueta
if 'spike' not in df.columns:
    raise ValueError("[❌ ERROR] La columna 'spike' no está presente en el CSV")

# Variables predictoras
columnas_usadas = ['fuerza_cuerpo', 'score', 'rsi', 'momentum', 'spread', 'fuerza_mecha', 'bajistas']
X = df[columnas_usadas]
y = df['spike']

# Balanceo de clases
df_balanceado = pd.concat([resample(df[df.spike == 1],
                                    replace=True,
                                    n_samples=len(df[df.spike == 0]),
                                    random_state=42),
                           df[df.spike == 0]])

# Redefinir X e y con datos balanceados
X_balanceado = df_balanceado[columnas_usadas]
y_balanceado = df_balanceado['spike']

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_balanceado)

# División para test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_balanceado, test_size=0.2, random_state=42)

# Entrenamiento del modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluación
y_pred = model.predict(X_test)
print("\n===== [📊 EVALUACIÓN RF BALANCEADO] =====")
print(classification_report(y_test, y_pred))

# Guardado
os.makedirs("modelos", exist_ok=True)
joblib.dump(model, "modelos/model_spike.pkl")
joblib.dump(scaler, "modelos/scaler_rf_2.pkl")
print("\n[✅ MODELO GUARDADO] modelos/model_spike.pkl")
print("[✅ SCALER GUARDADO] modelos/scaler_rf_2.pkl")
