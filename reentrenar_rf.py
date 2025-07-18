# reentrenar_rf.py

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from utils.logs import log

# ========================
# Cargar experiencias
# ========================
df = pd.read_csv("data/experiencias_drl.csv")

# Columnas requeridas
columnas = ['score', 'futuro', 'bajistas', 'visual_spike', 'rf_spike', 'lstm_spike',
            'ema_diff', 'rsi', 'momentum', 'spread', 'monto', 'multiplicador', 'ganancia_estim', 'direccion']

# Verificación
faltantes = set(columnas) - set(df.columns)
if faltantes:
    raise Exception(f"❌ Columnas faltantes: {faltantes}")

# Limpieza de NaN
df.dropna(subset=columnas, inplace=True)

# ========================
# Entrenamiento GANANCIA
# ========================
X_g = df[['score', 'futuro', 'bajistas', 'visual_spike', 'rf_spike', 'lstm_spike', 
          'ema_diff', 'rsi', 'momentum', 'spread', 'monto', 'multiplicador']]
y_g = df['ganancia_estim']

X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_g, y_g, test_size=0.2, random_state=42)
modelo_g = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_g.fit(X_train_g, y_train_g)

preds_g = modelo_g.predict(X_test_g)
log(f"📈 GANANCIA RF - MSE: {mean_squared_error(y_test_g, preds_g):.4f}")
joblib.dump(modelo_g, "modelos/model_rf_ganancia.pkl")
log("✅ Modelo de ganancia guardado: model_rf_ganancia.pkl")

# ========================
# Entrenamiento DIRECCIÓN
# ========================
X_d = X_g.copy()
y_d = df['direccion'].astype(int)

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.2, random_state=42)
modelo_d = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_d.fit(X_train_d, y_train_d)

preds_d = modelo_d.predict(X_test_d)
log(f"📊 DIRECCIÓN RF - Accuracy: {accuracy_score(y_test_d, preds_d):.4f}")
joblib.dump(modelo_d, "modelos/model_rf_direccion.pkl")
log("✅ Modelo de dirección guardado: model_rf_direccion.pkl")
