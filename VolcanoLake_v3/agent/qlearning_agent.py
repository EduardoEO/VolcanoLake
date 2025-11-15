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
        # Q-table: estados x acciones (625 x 8 para mapa 25x25)
        self.q_values = np.zeros((env.observation_space.n, env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []
    
    def get_action(self, env, obs: int) -> int:
        """
        Selecciona acción usando política epsilon-greedy.
        
        Args:
            env: Entorno
            obs: Estado actual (entero de 0 a 624, predeterminado)
        
        Returns:
            Acción seleccionada (entero de 0 a 7)
        """
        if np.random.random() < self.epsilon:
            # Exploración: acción aleatoria (0-7)
            return env.action_space.sample()
        else:
            # Explotación: mejor acción conocida
            return int(np.argmax(self.q_values[obs]))

    def update(self, obs, action, reward, terminated, next_obs):
        """
        Actualiza Q-table usando Q-Learning.
        
        Q(s,a) <- Q(s,a) + alpha[r + gamma*max Q(s',a') - Q(s,a)]
        """
        # Si el episodio terminó, no hay futuro por lo que será 0 y si no terminó será 1. Por lo tanto: 1. Si no terminó, future_q_value = max_a' Q(s', a') | 2. Si terminó, future_q_value = 0
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        
        # Calcular TD error
        td_error = reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        
        # Actualizar Q-value
        self.q_values[obs][action] += self.lr * td_error
        
        # Guardar error para análisis
        self.training_error.append(td_error)

    def decay_epsilon(self):
        """Reduce epsilon linealmente hasta el mínimo."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)