# core/main_control.py

import os
import sys
from utils.logs import log

def detener_bot():
    log("🛑 BOT DETENIDO MANUALMENTE POR GUARDIAN DE PÉRDIDAS")
    os._exit(0)
