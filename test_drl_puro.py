# test_drl_puro.py

import pandas as pd
from core.contexto import construir_contexto
from core.ia_drl import procesar_decision_drl

# Ruta al CSV con contexto real ya procesado por el bot
CSV = "data/contexto_historico.csv"

try:
    df = pd.read_csv(CSV)
    print(f"📄 CSV cargado con {len(df)} filas.")
except Exception as e:
    print(f"[❌ ERROR] No se pudo cargar el CSV: {e}")
    exit()

if len(df) < 60:
    print(f"[⛔ INSUFICIENTE] Se requieren al menos 60 velas para test. Solo hay {len(df)}.")
    exit()

df = df.tail(120)
contexto = construir_contexto(df, cantidad=60)

if contexto is None:
    print("[❌ CONTEXTO] Error al construir contexto.")
    exit()

# Simular condiciones realistas
capital = 500.0
multiplicadores = [100, 200, 300, 400]

# Ejecutar decisión DRL
print("🔍 Evaluando decisión DRL...")
decision = procesar_decision_drl(contexto, capital, multiplicadores)

# Extraer resultados
score_drl = decision.get("score", 0.0)
accion = decision.get("accion", 0)
monto = decision.get("monto", 0)
mult = decision.get("multiplicador", 0)
ganancia = decision.get("ganancia_esperada", 0.0)
duracion = decision.get("duracion_estimada", 0)
permitir = decision.get("permitir_entrada", False)

# Mostrar resultados
print("\n📊 RESULTADO DECISIÓN DRL:")
print(f"📌 Acción: {accion}")
print(f"📈 Score DRL: {score_drl:.4f}")
print(f"💰 Monto sugerido: ${monto}")
print(f"🎯 Multiplicador: {mult}")
print(f"💡 Ganancia estimada: ${ganancia}")
print(f"⏳ Duración estimada: {duracion} segundos")
print(f"🚦 Entrada permitida: {'Sí' if permitir else 'No'}")
