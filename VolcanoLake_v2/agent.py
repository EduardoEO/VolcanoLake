import numpy as np

class VolcanoLakeAgent:
    """
    Agente Q-Learning para el entorno VolcanoLake (FrozenLake modificado).
    
    Implementa el algoritmo Q-Learning con política epsilon-greedy para exploración.
    Mantiene una Q-table que mapea estados y acciones a valores Q esperados.
    """
    def __init__(
        self,
        env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """
        Inicializa el agente Q-Learning con sus hiperparámetros.
        
        Args:
            env: Entorno de Gymnasium (debe tener observation_space y action_space)
            learning_rate (float): Tasa de aprendizaje alfa (controla velocidad de actualización)
            initial_epsilon (float): Valor inicial de exploración (típicamente 1.0 = 100%)
            epsilon_decay (float): Cantidad que se reduce epsilon después de cada episodio
            final_epsilon (float): Valor mínimo de epsilon (típicamente 0.1 = 10%)
            discount_factor (float): Factor de descuento gamma para recompensas futuras (default: 0.95)
        """
        self.q_values = np.zeros((env.observation_space.n, env.action_space.n)) # Se crea la Q-table con los estados y las acciones, siendo 16x4
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []

    # Con el método get_action te devuelve un valor 0-3 según la condición
    def get_action(self, env, obs: tuple[int, int, int]) -> int:
        """
        Selecciona una acción usando la política epsilon-greedy.
        
        Estrategia de exploración vs explotación:
        - Con probabilidad epsilon: acción aleatoria (EXPLORACIÓN)
        - Con probabilidad (1-epsilon): mejor acción conocida (EXPLOTACIÓN)
        
        Args:
            env: Entorno de Gymnasium (para muestrear acciones aleatorias)
            obs (int): Estado actual del agente (posición en el grid)
            
        Returns:
            int: Acción seleccionada (0=izquierda, 1=abajo, 2=derecha, 3=arriba)
        """
        if np.random.random() < self.epsilon: # Si el número generado es menor que el epsilon genera un numero aleatorio 0-3 (las posibles acciones). Por lo tanto, si el epsilon es muy alto más probabilidad hay de explorar y viceversa
            return env.action_space.sample()
        return int(np.argmax(self.q_values[obs])) # Si no se cumple la condición se toma el valor de 0-3 que más alto esté en dicha observación de nuestra tabla-Q

    # Con el método update se actualiza la Q-table y el historial de errores de entrenamiento.
    def update(self, obs, action, reward, terminated, next_obs):
        """
        Actualiza la Q-table usando la ecuación de Bellman de Q-Learning.
        
        Ecuación de actualización:
        Q(s,a) ← Q(s,a) + alfa x [r + gamma x max Q(s',a') - Q(s,a)]
                                    └─ TD Target ─┘  └─ Valor actual ─┘
                            └──────── TD Error ────────────┘
        
        Args:
            obs (int): Estado actual
            action (int): Acción tomada
            reward (float): Recompensa recibida
            terminated (bool): Si el episodio terminó
            next_obs (int): Siguiente estado observado
        """
        # ===== CÁLCULO DEL VALOR FUTURO =====
        # Si el episodio terminó: future_q_value = 0 (no hay futuro)
        # Si continúa: future_q_value = max_a' Q(s', a') (mejor valor posible en siguiente estado)
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        
        # ===== CÁLCULO DEL ERROR TEMPORAL (TD ERROR) =====
        # Mide la diferencia entre la predicción y la realidad
        # TD Error = (r + gamma x max Q(s',a')) - Q(s,a)
        #            └── TD Target ────┘   └─ Estimación actual ─┘
        td_error = reward + self.discount_factor * future_q_value - self.q_values[obs][action]

        # Interpretación del TD Error:
        # - TD Error > 0: La acción fue mejor de lo esperado -> aumenta Q
        # - TD Error < 0: La acción fue peor de lo esperado -> disminuye Q
        # - TD Error ≈ 0: La predicción fue correcta -> cambio mínimo

        # ===== ACTUALIZACIÓN DE LA Q-TABLE =====
        # Solo actualiza la celda Q(s,a) correspondiente al estado-acción actual
        # Nueva Q(s,a) = Vieja Q(s,a) + alfa x TD Error
        self.q_values[obs][action] += self.lr * td_error

        # ===== REGISTRO DEL ERROR PARA ANÁLISIS =====
        # Guarda el TD error para graficar la convergencia del entrenamiento
        self.training_error.append(td_error)

    def decay_epsilon(self):
        """
        Reduce el valor de epsilon linealmente para disminuir la exploración.
        
        Se llama al final de cada episodio para implementar el decaimiento
        lineal de la tasa de exploración. Asegura que epsilon nunca baje
        del valor mínimo configurado (final_epsilon).
        
        Fórmula: epsilon_nuevo = max(epsilon_final, epsilon_actual - decay)
        """
        # Reduce epsilon pero nunca por debajo del mínimo establecido
        # Esto mantiene algo de exploración incluso al final del entrenamiento
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)