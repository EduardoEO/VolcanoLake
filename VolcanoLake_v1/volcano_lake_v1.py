import gymnasium as gym
import numpy as np
from tqdm import tqdm
from utils import plot_training
# -----------------------------
# Agent block
# -----------------------------

# Ecuación de Bellman para Q-learning:
    # Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]
    # Donde:
        #   Q(s, a): self.q_values[estados][acciones]
        #   α (learning rate): self.lr
        #   r (reward): reward
        #   γ (discount factor): self.discount_factor
        #   max_a' Q(s', a'): np.max(self.q_values[next_obs])
        #   s: obs actual
        #   a: action tomada
        #   s': next_obs
        
# Ejemplo visual de Q-table para FrozenLake 4x4:
#      a0     a1     a2     a3
# s0  0.0   0.0   0.0   0.0
# s1  0.0   0.0   0.0   0.0
# s2  0.0   0.0   0.0   0.0
# ...
# s15 0.0   0.0   0.0   0.0
# Donde s0-s15 son los estados (filas) y a0-a3 las acciones (columnas: 0=izquierda, 1=abajo, 2=derecha, 3=arriba)

class VolcanoLakeAgent:
    def __init__(
        self,
        env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        self.q_values = np.zeros((env.observation_space.n, env.action_space.n)) # Se crea la Q-table con los estados y las acciones, siendo 16x4.
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []
    
    # Con el método get_action te devuelve un valor 0-3 según la condición.
    def get_action(self, env, obs: tuple[int, int, int]) -> int:
        if np.random.random() < self.epsilon: # Si el número generado es menor que el epsilon genera un numero aleatorio 0-3 (las posibles acciones). Por lo tanto, si el epsilon es muy alto más probabilidad hay de explorar y viceversa.
            return env.action_space.sample()
        return int(np.argmax(self.q_values[obs])) # Si no se cumple la condición se toma el valor de 0-3 que más alto esté en dicha observación de nuestra tabla-Q. 

    # Con el método update se actualiza la Q-table y el historial de errores de entrenamiento.
    # Ejemplo numérico simple de actualización Q-learning:
    #
    # Supón:
    #   obs = 2
    #   action = 1
    #   reward = 1
    #   next_obs = 3
    #   terminated = False
    #
    # En la tabla Q:
    #   Q(2, 1) = 0.5
    #   max_a' Q(3, a') = 0.8
    #
    # Con:
    #   lr = 0.1
    #   discount_factor = 0.9
    #
    # Cálculos:
    #   future_q_value = 0.8
    #   td_error = 1 + 0.9*0.8 - 0.5 = 1.22 - 0.5 = 0.72
    #   Q(2,1) = 0.5 + 0.1 * 0.72 = 0.572
    #
    # Resultado:
    #   La celda [2][1] aumenta de 0.5 → 0.572
    #   El agente ha aprendido que esa acción parece más prometedora.
    def update(self, obs, action, reward, terminated, next_obs):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs]) # Si el episodio terminó, no hay futuro por lo que será 0 y si no terminó será 1. Por lo tanto: 1. Si no terminó, future_q_value = max_a' Q(s', a') | 2. Si terminó, future_q_value = 0
        td_error = reward + self.discount_factor * future_q_value - self.q_values[obs][action] # Esto calcula el error temporal (TD error), que mide cuánto difiere la predicción anterior de la nueva observación real. TD error = (𝑟+𝛾max⁡𝑄(𝑠′,𝑎′))−𝑄(𝑠,𝑎)TD error=(r+γa′maxQ(s′,a′))−Q(s,a). Si el TD error es grande, significa que el agente aprendió algo nuevo e importante.Si es pequeño, significa que ya predijo bien la recompensa esperada.
        self.q_values[obs][action] += self.lr * td_error # Actualiza solo una celda específica de la Q-table: la correspondiente al estado actual y la acción que tomó. 𝑄(𝑠,𝑎)=𝑄(𝑠,𝑎)+𝛼×TD errorQ(s,a)=Q(s,a)+α×TDerror. Si la acción resultó mejor de lo esperado aumenta el valor Q.Si fue peor disminuye el valor Q. Con el tiempo, las acciones buenas mantienen valores Q altos.
        self.training_error.append(td_error) # Guarda el error TD de cada actualización, para analizarlo (por ejemplo, graficar cómo va bajando el error medio a lo largo del entrenamiento).

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay) # El epsilon se reduce de manera lineal

  
def train_volcanoLake_agent(n_episodes):
    env = gym.make("FrozenLake-v1", is_slippery = True) # Inicializa el entorno con casillas deslizantes. Esto hace que le cueste más al agente aprender una ruta óptima.
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes) # Registro de estadísticas de cada episodio

    learning_rate = 0.01
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.1

    agent = VolcanoLakeAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    for _ in tqdm(range(n_episodes)):
        obs, info = env.reset() # Reinicio del entrono después de cada episodio
        done = False
        while not done: # Un bucle del while equivale a un episodio
            action = agent.get_action(env, obs) # Elige la acción según la política definida (epsilon-greedy)
            next_obs, reward, terminated, truncated, info = env.step(action) # Ejecuta la acción "rellenando" las variables.
            agent.update(obs, action, reward, terminated, next_obs) # Actualiza la Q-table.
            obs = next_obs # Actualización del estado actual.
            done = terminated or truncated # Fin del episodio.
        agent.decay_epsilon() # Reducir el epsilon después de cada episodio.

    return env, agent

# -----------------------------
# Main block
# -----------------------------

if __name__ == "__main__":
    plot_save = True
    n_episodes = 100_000
    
    env, agent = train_volcanoLake_agent(n_episodes)
    plot_training(env, agent, plot_save)
    
    np.set_printoptions(precision=2, suppress=True)
    print("\nQ-table final (filas=estados, columnas=acciones):\n")
    print(agent.q_values)
    
    print("\nMejor acción por estado (0=izq, 1=abajo, 2=dcha, 3=arriba):")
    best_actions = np.argmax(agent.q_values, axis=1).reshape(4, 4)    
    print(best_actions)
    
    env.close() # Close the environment