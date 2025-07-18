import pickle
import pandas as pd

# Ruta al archivo columnas_drl.pkl
RUTA_COLUMNAS_PKL = "modelos/columnas_drl.pkl"
# Ruta al dataset real que genera el contexto
RUTA_DATASET = "data/dataset_drl.csv"

try:
    # Cargar columnas esperadas por el modelo DRL
    with open(RUTA_COLUMNAS_PKL, "rb") as f:
        columnas_modelo = pickle.load(f)
    print(f"\n✅ Columnas esperadas por el modelo DRL ({len(columnas_modelo)}):\n{columnas_modelo}")

    # Cargar DataFrame del dataset real
    df = pd.read_csv(RUTA_DATASET)
    columnas_dataset = df.columns.tolist()
    print(f"\n📄 Columnas encontradas en el dataset actual ({len(columnas_dataset)}):\n{columnas_dataset}")

    # Comparar listas
    columnas_faltantes = [col for col in columnas_modelo if col not in columnas_dataset]
    columnas_extra = [col for col in columnas_dataset if col not in columnas_modelo]

    if not columnas_faltantes and not columnas_extra:
        print("\n✅✅ Las columnas del dataset coinciden perfectamente con las que espera el modelo DRL.")
    else:
        print("\n⚠️ INCONSISTENCIA DETECTADA:")
        if columnas_faltantes:
            print(f"❌ Faltan columnas requeridas por el modelo: {columnas_faltantes}")
        if columnas_extra:
            print(f"⚠️ Columnas adicionales que no se usan en el modelo: {columnas_extra}")

except Exception as e:
    print(f"\n❌ Error en la verificación: {e}")
