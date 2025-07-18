# core/drl_agente.py

import numpy as np
import tensorflow as tf
import joblib
import os
from utils.logs import log
from collections import deque
import random

# Ruta modelo y scaler
MODELO_PATH = "modelos/drl_model.h5"
SCALER_PATH = "modelos/columnas_drl.pkl"

class DRLAgente:
    def __init__(self):
        self.modelo = self.cargar_modelo()
        self.experiencias = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 0.5  # Puede ir decayendo
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.tamano_estado = 10
        self.tamano_accion = 4
        self.batch_size = 32

    def cargar_modelo(self):
        try:
            if os.path.exists(MODELO_PATH):
                modelo = tf.keras.models.load_model(MODELO_PATH)
                log("✅ Modelo DRL cargado correctamente.")
                return modelo
            else:
                modelo = tf.keras.Sequential([
                    tf.keras.layers.Dense(64, input_dim=10, activation="relu"),
                    tf.keras.layers.Dense(32, activation="relu"),
                    tf.keras.layers.Dense(4, activation="linear"),
                ])
                modelo.compile(optimizer="adam", loss="mse")
                log("⚠️ Modelo DRL creado nuevo desde cero.")
                return modelo
        except Exception as e:
            log(f"[❌ ERROR CARGA DRL] {e}")
            return None

    def actuar(self, estado):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, self.tamano_accion - 1)
        q_values = self.modelo.predict(estado, verbose=0)
        return np.argmax(q_values[0])

    def guardar(self, estado, accion, recompensa, siguiente_estado, terminado):
        self.experiencias.append((estado, accion, recompensa, siguiente_estado, terminado))

    def entrenar(self):
        if len(self.experiencias) < self.batch_size:
            return

        minibatch = random.sample(self.experiencias, self.batch_size)

        for estado, accion, recompensa, siguiente_estado, terminado in minibatch:
            objetivo = recompensa
            if not terminado:
                objetivo = recompensa + self.gamma * np.amax(self.modelo.predict(siguiente_estado, verbose=0)[0])

            objetivo_futuro = self.modelo.predict(estado, verbose=0)
            objetivo_futuro[0][accion] = objetivo

            self.modelo.fit(estado, objetivo_futuro, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def guardar_modelo(self):
        try:
            self.modelo.save(MODELO_PATH)
        except Exception as e:
            log(f"[❌ ERROR GUARDAR MODELO DRL] {e}")

# Instancia global
drl_agent = DRLAgente()
