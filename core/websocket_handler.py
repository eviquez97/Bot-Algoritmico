# test_inicio_basico.py
from modelos.cargar_modelos import cargar_todos_los_modelos
from config import SYMBOL

print("✅ TEST DE INICIO BOT MONSTRUO")
print(f"📈 SYMBOL: {SYMBOL}")

# Cargar modelos IA
modelos = cargar_todos_los_modelos()

# Verificar que todos los modelos se cargaron correctamente
for nombre, modelo in modelos.items():
    if modelo:
        print(f"[✅ MODELO CARGADO] {nombre}")
    else:
        print(f"[❌ MODELO FALTANTE] {nombre}")

