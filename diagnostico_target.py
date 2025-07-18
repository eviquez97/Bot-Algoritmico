import os

TARGET_PATRONES = [
    '["target"]',
    "['target']",
    'columns=["target"]',
    "columns=['target']",
    '"target"',
    "'target'"
]

def contiene_target(linea):
    return any(pat in linea and not linea.strip().startswith("#") for pat in TARGET_PATRONES)

def escanear_archivos_en_busca_de_target(base_dir="."):
    print("🔍 Iniciando escaneo de archivos para encontrar uso de 'target'...\n")
    errores_detectados = False

    for root, _, files in os.walk(base_dir):
        for nombre_archivo in files:
            if nombre_archivo.endswith(".py"):
                ruta = os.path.join(root, nombre_archivo)
                try:
                    with open(ruta, "r", encoding="utf-8") as f:
                        for i, linea in enumerate(f, start=1):
                            if contiene_target(linea):
                                if "drop" in linea or "[" in linea:
                                    print(f"⚠️ Posible acceso conflictivo a 'target' en {ruta} | Línea {i}:\n    {linea.strip()}\n")
                                    errores_detectados = True
                except Exception as e:
                    print(f"❌ Error al leer {ruta}: {e}")

    if not errores_detectados:
        print("✅ No se encontraron referencias problemáticas a 'target'.")
    else:
        print("⚠️ Revisa los archivos listados arriba. Puede que uno esté intentando acceder a 'target' en producción.")

# Ejecutar diagnóstico
if __name__ == "__main__":
    escanear_archivos_en_busca_de_target()
