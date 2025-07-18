from sklearn.ensemble import RandomForestClassifier
import numpy as np

# CARGA DEL MODELO (ya debe estar cargado globalmente)
from modelos.cargador_modelos import model_spike, model_lstm_spike, scaler_rf_2

# BLOQUEO SPIKE ELITE — FUSIÓN DE PREDICCIONES
def bloqueo_spike_elite(df):
    try:
        columnas_esperadas = ['open', 'high', 'low', 'close', 'spread', 'momentum', 'ema', 'rsi']
        X_rf = df[columnas_esperadas].tail(30)
        X_rf = scaler_rf_2.transform(X_rf)
        pred_rf = model_spike.predict(X_rf)
        spike_rf = int(np.mean(pred_rf) > 0.7)

        X_lstm = df[columnas_esperadas].tail(30).values.reshape(1, 30, len(columnas_esperadas))
        prob_lstm = model_lstm_spike.predict(X_lstm)[0][0]
        spike_lstm = prob_lstm > 0.7

        return spike_rf == 1 or spike_lstm

    except Exception as e:
        print(f"[❌ ERROR BLOQUEO SPIKE ELITE] {type(e).__name__} - {e}")
        return False
