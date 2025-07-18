import joblib

# Ruta al archivo columnas_drl.pkl
ruta = "modelos/columnas_drl.pkl"

# Cargar y mostrar las columnas
try:
    columnas = joblib.load(ruta)
    print("📄 COLUMNAS DRL:")
    print(columnas)
except Exception as e:
    print(f"[❌ ERROR] No se pudo cargar el archivo: {e}")
