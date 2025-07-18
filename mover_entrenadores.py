import os
import shutil

# Carpeta destino
DESTINO = "entrenamiento_offline"
os.makedirs(DESTINO, exist_ok=True)

# Lista de scripts a mover
scripts_entrenadores = [
    "entrenador_spike_rf.py",
    "entrenador_lstm_spike.py",
    "entrenar_modelo_cierre.py",
    "entrenar_modelo_direccion_rf.py",
    "entrenar_modelo_lstm_spike.py",
    "preparar_dataset_spike.py",
    "limpiar_dataset_spike.py",
    "generar_dataset_spike_moderno.py",
    "generar_dataset_entrenamiento.py",
    "verificador_spike.py",
]

# Mover scripts si existen
for script in scripts_entrenadores:
    if os.path.exists(script):
        destino_path = os.path.join(DESTINO, script)
        shutil.move(script, destino_path)
        print(f"✅ Movido: {script} -> {destino_path}")
    else:
        print(f"⚠️ No encontrado: {script}")

print("🎓 Todos los scripts de entrenamiento fueron movidos correctamente.")
