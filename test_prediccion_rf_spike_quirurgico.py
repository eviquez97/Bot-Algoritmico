import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix

# Cargar dataset
RUTA_CSV = "data/dataset_spike_monstruo_limpio.csv"
df = pd.read_csv(RUTA_CSV)

# Aseguramos que esté la columna de etiqueta
if 'spike_anticipado' not in df.columns:
    print("❌ El dataset no contiene la columna 'spike_anticipado'.")
    exit()

# Cargar modelo y columnas
with open("modelos/model_spike.pkl", "rb") as f:
    model = pickle.load(f)

columnas_modelo = [
    'fuerza_cuerpo', 'fuerza_mecha', 'bajistas', 'rsi',
    'momentum', 'spread', 'score', 'ema', 'variacion'
]

# Validar columnas
for col in columnas_modelo:
    if col not in df.columns:
        print(f"❌ Falta columna: {col}")
        exit()

# Preparar los datos
df_filtrado = df.dropna(subset=columnas_modelo + ['spike_anticipado']).tail(200)
X = df_filtrado[columnas_modelo]
y_true = df_filtrado['spike_anticipado']

# Predicción
y_pred_prob = model.predict_proba(X)[:, 1]
y_pred_binaria = (y_pred_prob >= 0.5).astype(int)

# Imprimir comparación línea por línea
print("\n🔬 Comparación de últimas 200 filas:")
print("Fila | Real | Predicción RF")
for i in range(len(y_true)):
    real = y_true.iloc[i]
    pred = round(y_pred_prob[i], 2)
    print(f"{df_filtrado.index[i]} |   {real}   |     {pred}")

# Métricas
print("\n📊 Reporte de clasificación (RF spike anticipado):")
print(classification_report(y_true, y_pred_binaria, digits=2))

print("📉 Matriz de confusión:")
print(confusion_matrix(y_true, y_pred_binaria))
