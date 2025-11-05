import numpy as np

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
    def update(self, obs, action, reward, terminated, next_obs):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs]) # Si el episodio terminó, no hay futuro por lo que será 0 y si no terminó será 1. Por lo tanto: 1. Si no terminó, future_q_value = max_a' Q(s', a') | 2. Si terminó, future_q_value = 0
        td_error = reward + self.discount_factor * future_q_value - self.q_values[obs][action] # Esto calcula el error temporal (TD error), que mide cuánto difiere la predicción anterior de la nueva observación real. TD error = (𝑟+𝛾max⁡𝑄(𝑠′,𝑎′))−𝑄(𝑠,𝑎)TD error=(r+γa′maxQ(s′,a′))−Q(s,a). Si el TD error es grande, significa que el agente aprendió algo nuevo e importante.Si es pequeño, significa que ya predijo bien la recompensa esperada.
        self.q_values[obs][action] += self.lr * td_error # Actualiza solo una celda específica de la Q-table: la correspondiente al estado actual y la acción que tomó. 𝑄(𝑠,𝑎)=𝑄(𝑠,𝑎)+𝛼×TD errorQ(s,a)=Q(s,a)+α×TDerror. Si la acción resultó mejor de lo esperado aumenta el valor Q.Si fue peor disminuye el valor Q. Con el tiempo, las acciones buenas mantienen valores Q altos.
        self.training_error.append(td_error) # Guarda el error TD de cada actualización, para analizarlo (por ejemplo, graficar cómo va bajando el error medio a lo largo del entrenamiento).

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay) # El epsilon se reduce de manera lineal
