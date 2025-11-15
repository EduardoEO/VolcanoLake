import gymnasium as gym
import numpy as np
from tqdm import tqdm

from utils import plot_training

# ========================================
# FUNDAMENTOS TEÓRICOS: Q-LEARNING
# ========================================

# Ecuación de Bellman para Q-learning:
#   Q(s, a) <- Q(s, a) + alfa [r + gamma max_a' Q(s', a') - Q(s, a)]
#
# Componentes de la ecuación:
#   - Q(s, a): Valor Q actual para el estado s y acción a
#   - alfa (learning rate): Controla qué tan rápido aprende (0-1)
#   - r (reward): Recompensa inmediata recibida
#   - gamma (discount factor): Importancia de recompensas futuras (0-1)
#   - max_a' Q(s', a'): Mejor valor Q posible en el siguiente estado
#   - s: Estado actual (obs)
#   - a: Acción tomada (action)
#   - s': Siguiente estado (next_obs)

# Interpretación:
#   El valor Q se ajusta según la diferencia entre lo que esperábamos
#   y lo que realmente obtuvimos (TD error = temporal difference error)

# ========================================
# ESTRUCTURA DE LA Q-TABLE
# ========================================

# Ejemplo visual de Q-table para FrozenLake 4x4:
#
#        LEFT   DOWN   RIGHT  UP
#        (a0)   (a1)   (a2)   (a3)
# s0     0.0    0.0    0.0    0.0
# s1     0.0    0.0    0.0    0.0
# s2     0.0    0.0    0.0    0.0
# ...    ...    ...    ...    ...
# s15    0.0    0.0    0.0    0.0
#
# Donde:
#   - Filas (s0-s15): Los 16 estados del grid 4x4
#   - Columnas (a0-a3): Las 4 acciones posibles
#   - Valores: Recompensa esperada al tomar acción a en estado s

# ========================================
# AGENTE Q-LEARNING
# ========================================

class VolcanoLakeAgent:
    """
    Agente Q-Learning para el entorno VolcanoLake (FrozenLake modificado).
    
    Implementa el algoritmo Q-Learning con política epsilon-greedy para
    encontrar la política óptima de navegación en un lago congelado.
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
            env: Entorno de Gymnasium
            learning_rate (float): alfa - Tasa de aprendizaje (típicamente 0.01-0.1)
            initial_epsilon (float): epsilon inicial - Exploración al inicio (típicamente 1.0)
            epsilon_decay (float): Reducción de ε por episodio
            final_epsilon (float): epsilon mínimo - Exploración residual (típicamente 0.1)
            discount_factor (float): gamma - Descuento de recompensas futuras (default: 0.95)
        """
        self.q_values = np.zeros((env.observation_space.n, env.action_space.n)) # Se crea la Q-table con los estados y las acciones, siendo 16x4.
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []
    
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

    def update(self, obs, action, reward, terminated, next_obs):
        """
        Actualiza la Q-table usando la ecuación de Bellman.
        
        Proceso de actualización:
        1. Calcula el mejor valor futuro posible
        2. Calcula el error TD (diferencia entre expectativa y realidad)
        3. Ajusta el valor Q en proporción al error
        
        Ejemplo numérico:
        ---------------
        Situación:
          - Estado actual: s=2
          - Acción tomada: a=1 (abajo)
          - Recompensa: r=1
          - Siguiente estado: s'=3
          - Q(2,1) actual = 0.5
          - max_a' Q(3,a') = 0.8
          - lr = 0.1, gamma = 0.9
        
        Cálculos:
          - future_q_value = 0.8
          - td_error = 1 + 0.9x0.8 - 0.5 = 1.22
          - Nuevo Q(2,1) = 0.5 + 0.1x1.22 = 0.622
        
        Resultado:
          La celda [2][1] aumenta de 0.5 → 0.622
          El agente aprendió que esa acción es mejor de lo que pensaba
        
        Args:
            obs (int): Estado actual
            action (int): Acción tomada
            reward (float): Recompensa recibida
            terminated (bool): Si el episodio terminó
            next_obs (int): Siguiente estado
        """
        # ===== CÁLCULO DEL VALOR FUTURO =====
        # Si el episodio terminó: future_q_value = 0 (no hay futuro)
        # Si continúa: future_q_value = max_a' Q(s', a') (mejor valor posible en siguiente estado)
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        
        # ===== PASO 2: CALCULAR ERROR TEMPORAL (TD ERROR) =====
        # Mide cuánto nos equivocamos en nuestra predicción
        # TD error = (Recompensa real + Valor futuro) - Predicción anterior
        td_error = reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        
        # Interpretación del TD error:
        # - TD > 0: La acción fue MEJOR de lo esperado -< aumenta Q
        # - TD < 0: La acción fue PEOR de lo esperado -> disminuye Q
        # - TD ~= 0: La predicción fue CORRECTA -> cambio mínimo
        
        # ===== PASO 3: ACTUALIZAR Q-TABLE =====
        # Solo modifica la celda Q(s,a) correspondiente
        # Actualización proporcional al error: más error = más cambio
        self.q_values[obs][action] += self.lr * td_error
        
        # ===== PASO 4: REGISTRAR ERROR PARA ANÁLISIS =====
        # Guarda el TD error para monitorear el aprendizaje
        # Un error decreciente indica que el agente está convergiendo
        self.training_error.append(td_error)

    def decay_epsilon(self):
        """
        Reduce epsilon linealmente para disminuir la exploración.
        
        Se ejecuta al final de cada episodio. A medida que el agente
        aprende, explora menos y explota más su conocimiento.
        
        Ejemplo de evolución típica:
          Episodio 0:     epsilon = 1.0   (100% exploración)
          Episodio 5000:  epsilon = 0.55  (55% exploración)
          Episodio 10000: epsilon = 0.1   (10% exploración, se mantiene)
        """
        # Reduce epsilon pero nunca por debajo del mínimo
        # Mantiene exploración residual para descubrir cambios
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

# ========================================
# FUNCIÓN DE ENTRENAMIENTO
# ========================================

def train_volcanoLake_agent(n_episodes):
    """
    Entrena un agente Q-Learning en el entorno FrozenLake.
    
    Configuración del entorno:
    • FrozenLake-v1: Grid 4x4 con casillas deslizantes (is_slippery=True)
    • Casillas deslizantes simulan hielo resbaladizo (realismo)
    • Hace más difícil el aprendizaje pero más robusto
    
    Args:
        n_episodes (int): Número de episodios de entrenamiento
        
    Returns:
        tuple: (env, agent) - Entorno y agente entrenado
    """
    # ===== CONFIGURACIÓN DEL ENTORNO =====
    # Crea entorno FrozenLake predeterminado
    env = gym.make("FrozenLake-v1", is_slippery = True)
    
    # Wrapper para registrar estadísticas por episodio
    # Guarda recompensas acumuladas y duración de episodios
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    # ===== CONFIGURACIÓN DE HIPERPARÁMETROS =====
    learning_rate = 0.01
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.1

    # ===== INICIALIZACIÓN DEL AGENTE =====
    agent = VolcanoLakeAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    # ===== BUCLE DE ENTRENAMIENTO =====
    # Itera por n_episodes con barra de progreso
    for _ in tqdm(range(n_episodes)):
        obs, info = env.reset() # Reinicio del entrono después de cada episodio
        done = False
        # Cada iteración equivale un paso del agente
        while not done:
            action = agent.get_action(env, obs) # Elige la acción según la política definida (epsilon-greedy)
            next_obs, reward, terminated, truncated, info = env.step(action) # Ejecuta la acción "rellenando" las variables
            agent.update(obs, action, reward, terminated, next_obs) # Actualiza la Q-table
            obs = next_obs # Actualización del estado actual
            done = terminated or truncated # Fin del episodio
        agent.decay_epsilon() # Reducir el epsilon después de cada episodio

    return env, agent

# ========================================
# BLOQUE PRINCIPAL DE EJECUCIÓN
# ========================================

if __name__ == "__main__":
    # ===== CONFIGURACIÓN =====
    plot_save = True
    n_episodes = 100_000
    
    # ===== ENTRENAMIENTO =====
    env, agent = train_volcanoLake_agent(n_episodes)
    
    # ===== VISUALIZACIÓN DE RESULTADOS =====
    plot_training(env, agent, plot_save)
    
    # ===== ANÁLISIS DE LA Q-TABLE =====
    # Configuración para imprimir arrays de forma legible
    np.set_printoptions(precision=2, suppress=True)
    print("\nQ-table final (filas=estados, columnas=acciones):\n")
    print(agent.q_values)
    
    print("\nMejor acción por estado (0=izq, 1=abajo, 2=dcha, 3=arriba):")
    best_actions = np.argmax(agent.q_values, axis=1).reshape(4, 4)    
    print(best_actions)
    
    env.close() # Cerrar el entorno