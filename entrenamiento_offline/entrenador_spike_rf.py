import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Crear datos sintéticos compatibles con las columnas esperadas
n = 500
np.random.seed(42)

df = pd.DataFrame({
    'open': np.random.uniform(10000, 20000, n),
    'high': np.random.uniform(10000, 20000, n),
    'low': np.random.uniform(10000, 20000, n),
    'close': np.random.uniform(10000, 20000, n),
    'spread': np.random.uniform(0.1, 2.0, n),
    'momentum': np.random.uniform(-1.0, 1.0, n),
    'ema': np.random.uniform(10000, 20000, n),
    'rsi': np.random.uniform(0, 100, n),
    'target': np.random.randint(0, 2, n)
})

# Separar features y target
X = df.drop(columns=['target'])
y = df['target']

# Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Guardar modelo y scaler
joblib.dump(model, 'modelos/model_spike_rf.pkl')
joblib.dump(scaler, 'modelos/scaler_rf_2.pkl')

print("✅ Modelo y scaler guardados como 'model_spike_rf.pkl' y 'scaler_rf_2.pkl'")
