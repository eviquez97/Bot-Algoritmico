# utils/velas.py

from datetime import datetime

def construir_vela_desde_ticks(ticks):
    if not ticks:
        return None

    ticks_ordenados = sorted(ticks, key=lambda t: t['epoch'])

    open_price = float(ticks_ordenados[0]['quote'])
    close_price = float(ticks_ordenados[-1]['quote'])
    high_price = max(float(t['quote']) for t in ticks_ordenados)
    low_price = min(float(t['quote']) for t in ticks_ordenados)
    volumen = len(ticks_ordenados)
    timestamp = ticks_ordenados[0]['epoch']  # inicio del minuto

    return {
        "timestamp": timestamp,
        "datetime": datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
        "open": open_price,
        "close": close_price,
        "high": high_price,
        "low": low_price,
        "volumen": volumen
    }
