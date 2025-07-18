import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Ruta al CSV limpio
CSV_PATH = "data/dataset_spike_monstruo_limpio.csv"

# Cargar el dataset
df = pd.read_csv(CSV_PATH)

# Verificación de columnas esperadas
columnas_requeridas = ['fuerza_cuerpo', 'fuerza_mecha', 'bajistas', 'rsi', 'momentum', 'spread', 'score', 'ema', 'variacion', 'spike']
for col in columnas_requeridas:
    if col not in df.columns:
        raise ValueError(f"❌ ERROR: Falta la columna requerida: {col}")

# Eliminar filas con NaN
df = df.dropna()

# Separar features y target
X = df.drop(columns=['spike'])
y = df['spike']

# Escalar las features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División entrenamiento / validación
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Entrenar el modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluación
y_pred = model.predict(X_test)
print("\n📊 Clasificación:")
print(classification_report(y_test, y_pred))
print(f"🎯 Precisión: {accuracy_score(y_test, y_pred):.4f}")

# Guardar modelo y scaler
with open("modelos/model_spike.pkl", "wb") as f:
    pickle.dump(model, f)

with open("modelos/scaler_rf_2.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Modelo Random Forest SPIKE entrenado y guardado correctamente.")
