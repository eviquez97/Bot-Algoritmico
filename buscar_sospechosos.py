import os

# Directorio raíz del proyecto
directorio_base = os.getcwd()

# Palabras clave sospechosas
palabras_clave = [
    "from core.ia_drl_v2",
    "from core.ia_drl_agente",
    "from core.drl_estado",
    "objetivo_diario_alcanzado",
    "permitir_entrada = False",
    "monto = 0",
    "ganancia = 0",
    "multiplicador = 0",
    "return None"
]

print("\n🔍 Buscando uso de lógica antigua o bloqueos silenciosos...\n")

# Recorrer todos los archivos del proyecto
for root, _, files in os.walk(directorio_base):
    for file in files:
        if file.endswith(".py"):
            ruta_completa = os.path.join(root, file)
            with open(ruta_completa, "r", encoding="utf-8", errors="ignore") as f:
                lineas = f.readlines()
                for i, linea in enumerate(lineas):
                    for palabra in palabras_clave:
                        if palabra in linea:
                            print(f"⚠️ Encontrado '{palabra}' en {file} (Línea {i+1}): {linea.strip()}")

print("\n✅ Búsqueda finalizada.")
