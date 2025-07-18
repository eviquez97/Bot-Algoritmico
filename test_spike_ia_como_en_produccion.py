import pandas as pd
from core.ia_spike import evaluar_spike_ia

# Cargar CSV real con al menos 60 filas válidas
df = pd.read_csv("data/contexto_historico.csv")

# Solo si quieres forzar las últimas 60 filas:
df = df.tail(60)

# Ejecutar evaluación de Spike IA como lo hace el bot
evaluar_spike_ia(df)
