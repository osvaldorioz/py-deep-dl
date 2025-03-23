from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
from typing import List
import matplotlib
import matplotlib.pyplot as plt
import deep_rl
import json

matplotlib.use('Agg')  # Usar backend no interactivo
app = FastAPI()

def generate_state():
    return np.array([[0.1], [0.2], [0.3], [0.4]], dtype=np.float64)  # Estado como matriz columna explícita

def reward_function(state, action):
    return float(state[0, 0] + float(action))  # Asegurar compatibilidad con float


# Definir el modelo para el vector
class VectorF(BaseModel):
    vector: List[float]
    
@app.post("/deep-reinforcement-learning")
def calculo(state_dim: int, action_dim: int, episodes: int, alpha: float, gamma: float, eps: float, eps_decay: float):
    output_file = 'deep_reinforcement_learning.png'

    # Parámetros del entorno y el agente
    #state_dim = 4   Dimensión del estado
    #action_dim = 2  Número de acciones
    #episodes = 500
    #alpha = 0.01
    #gamma = 0.99
    #epsilon = 1.0
    #epsilon_decay = 0.995

    # Crear el agente
    agent = deep_rl.DeepQLearningAgent(state_dim, action_dim, alpha, gamma)

    # Entrenamiento
    rewards = []
    for episode in range(episodes):
        state = generate_state()
        total_reward = 0
        for _ in range(100):  # Máximo de 100 pasos por episodio
            state = np.ascontiguousarray(state, dtype=np.float64)  # Garantizar compatibilidad
            action = int(agent.select_action(state, float(eps)))  # forzar el cast de epsilon a float explícitamente
            next_state = generate_state()
            reward = reward_function(state, action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
        eps = max(eps * eps_decay, 0.01)  # Evitar que epsilon llegue a 0

    # Graficar las recompensas
    plt.plot(rewards)
    plt.xlabel('Episodios')
    plt.ylabel('Recompensa total')
    plt.title('Entrenamiento de Deep Q-Learning')
    #plt.show()

    plt.savefig(output_file)
    plt.close()
    
    j1 = {
        "Grafica": output_file
    }
    jj = json.dumps(str(j1))

    return jj

@app.get("/deep-reinforcement-learning-graph")
def getGraph(output_file: str):
    return FileResponse(output_file, media_type="image/png", filename=output_file)