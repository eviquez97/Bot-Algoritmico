import pandas as pd

def construir_contexto_drl(df, n=30):
    if len(df) < n:
        return None
    columnas = ['open', 'high', 'low', 'close', 'spread', 'momentum', 'ema', 'rsi']
    df_contexto = df[columnas].tail(n).copy()
    df_contexto.reset_index(drop=True, inplace=True)
    return df_contexto

