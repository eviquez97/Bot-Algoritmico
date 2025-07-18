import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from utils.logs import log

RUTA_DATASET = "data/dataset_cierre.csv"
RUTA_MODELO = "modelos/model_cierre.pkl"
RUTA_SCALER = "modelos/scaler_cierre.pkl"

def entrenar_modelo_cierre():
    try:
        log("♻️ [REENTRENAMIENTO CIERRE] Iniciando...")

        if not os.path.exists(RUTA_DATASET):
            log("❌ Dataset de cierre no encontrado. Creando base vacía...")
            df_base = pd.DataFrame(columns=['resultado', 'score', 'rsi', 'momentum', 'spread'])
            df_base.to_csv(RUTA_DATASET, index=False)
            return

        df = pd.read_csv(RUTA_DATASET).dropna()

        columnas_requeridas = ['resultado', 'score', 'rsi', 'momentum', 'spread']
        columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]
        if columnas_faltantes:
            log(f"❌ Columnas faltantes en dataset de cierre: {columnas_faltantes}")
            return

        X = df[['score', 'rsi', 'momentum', 'spread']]
        y = df['resultado'].astype(int)

        if len(X) < 30:
            log("⚠️ Dataset insuficiente para entrenar modelo de cierre (< 30 filas).")
            return

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X_train, y_train)

        joblib.dump(modelo, RUTA_MODELO)
        joblib.dump(scaler, RUTA_SCALER)

        log("✅ [MODELO CIERRE] Entrenamiento completado y archivos guardados correctamente.")

    except Exception as e:
        log(f"[❌ ERROR ENTRENAMIENTO CIERRE] {e}")


