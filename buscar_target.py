import os

RUTA_BASE = RUTA_BASE = "."

def buscar_target_en_archivos(ruta_base):
    print(f"🔍 Buscando referencias a 'target' en la carpeta: {ruta_base}")
    for root, _, files in os.walk(ruta_base):
        for archivo in files:
            if archivo.endswith(".py"):
                ruta = os.path.join(root, archivo)
                try:
                    with open(ruta, "r", encoding="utf-8") as f:
                        lineas = f.readlines()
                        for i, linea in enumerate(lineas):
                            if "target" in linea and not linea.strip().startswith("#"):
                                print(f"📌 {ruta} | Línea {i+1}: {linea.strip()}")
                except Exception as e:
                    print(f"❌ No se pudo leer {ruta}: {e}")

buscar_target_en_archivos(RUTA_BASE)
