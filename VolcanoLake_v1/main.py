# Autores: Eduardo Estefanía Ovejero y Álvaro Martín García 

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from utils import plot_training

# ========================================
# THEORETICAL FOUNDATIONS: Q-LEARNING
# ========================================

# Bellman equation for Q-learning:
#   Q(s, a) <- Q(s, a) + alpha [r + gamma max_a' Q(s', a') - Q(s, a)]
#
# Components of the equation:
#   - Q(s, a): Current Q-value for state s and action a
#   - alpha (learning rate): Controls how fast the agent learns (0-1)
#   - r (reward): Immediate reward received
#   - gamma (discount factor): Importance of future rewards (0-1)
#   - max_a' Q(s', a'): Best possible Q-value in the next state
#   - s: Current state (obs)
#   - a: Action taken (action)
#   - s': Next state (next_obs)

# Interpretation:
#   The Q-value is adjusted according to the difference between what
#   we expected and what we actually obtained (TD error = temporal difference error)

# ========================================
# Q-TABLE STRUCTURE
# ========================================

# Visual example of a Q-table for FrozenLake 4x4:
#
#        LEFT   DOWN   RIGHT  UP
#        (a0)   (a1)   (a2)   (a3)
# s0     0.0    0.0    0.0    0.0
# s1     0.0    0.0    0.0    0.0
# s2     0.0    0.0    0.0    0.0
# ...    ...    ...    ...    ...
# s15    0.0    0.0    0.0    0.0
#
# Where:
#   - Rows (s0-s15): The 16 states of the 4x4 grid
#   - Columns (a0-a3): The 4 possible actions
#   - Values: Expected reward when taking action a in state s

# ========================================
# Q-LEARNING AGENT
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
        # Create the Q-table with states and actions (e.g., 16x4)
        self.q_values = np.zeros((env.observation_space.n, env.action_space.n))
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
        # If a random number is less than epsilon, choose a random action
        # (exploration). High epsilon => more exploration.
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        # Otherwise choose the action with highest Q-value for the current observation
        return int(np.argmax(self.q_values[obs]))

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
        # ===== FUTURE VALUE CALCULATION =====
        # If the episode ended: future_q_value = 0 (no future)
        # If it continues: future_q_value = max_a' Q(s', a') (best possible value in next state)
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        
        # ===== STEP 2: CALCULATE TEMPORAL DIFFERENCE (TD) ERROR =====
        # Measures how wrong we were in our prediction
        # TD error = (Actual reward + Future value) - Previous prediction
        td_error = reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        
        # Interpretation of the TD error:
        # - TD > 0: The action was BETTER than expected -> increases Q
        # - TD < 0: The action was WORSE than expected -> decreases Q
        # - TD ~= 0: The prediction was CORRECT -> minimal change
        
        # ===== STEP 3: UPDATE Q-TABLE =====
        # Only modify the corresponding Q(s,a) cell
        # Update proportional to the error: larger error -> larger change
        self.q_values[obs][action] += self.lr * td_error
        
        # ===== STEP 4: RECORD ERROR FOR ANALYSIS =====
        # Save the TD error to monitor learning
        # A decreasing error indicates the agent is converging
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
        # Reduce epsilon but never below the minimum
        # Keeps a residual exploration chance to discover changes
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

# ========================================
# FUNCIÓN DE ENTRENAMIENTO
# ========================================

class VolcanoLakeAgent:
    """
    Q-Learning agent for the VolcanoLake environment (modified FrozenLake).
    
    Implements the Q-Learning algorithm with epsilon-greedy policy to
    find the optimal navigation policy in a frozen lake.
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
        Initializes the Q-Learning agent with its hyperparameters.
        
        Args:
            env: Gymnasium environment
            learning_rate (float): alpha - Learning rate (typically 0.01-0.1)
            initial_epsilon (float): initial epsilon - Exploration at start (typically 1.0)
            epsilon_decay (float): Reduction of epsilon per episode
            final_epsilon (float): minimum epsilon - Residual exploration (typically 0.1)
            discount_factor (float): gamma - Future reward discount (default: 0.95)
        """
        # Create the Q-table with states and actions (e.g., 16x4)
        self.q_values = np.zeros((env.observation_space.n, env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []
    
    def get_action(self, env, obs: tuple[int, int, int]) -> int:
        """
        Selects an action using the epsilon-greedy policy.
        
        Exploration vs exploitation strategy:
        - With probability epsilon: random action (EXPLORATION)
        - With probability (1-epsilon): best known action (EXPLOITATION)
        
        Args:
            env: Gymnasium environment (to sample random actions)
            obs (int): Agent's current state (grid position)
            
        Returns:
            int: Selected action (0=left, 1=down, 2=right, 3=up)
        """
        # If a random number is less than epsilon, choose a random action
        # (exploration). High epsilon => more exploration.
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        # Otherwise choose the action with highest Q-value for the current observation
        return int(np.argmax(self.q_values[obs]))

    def update(self, obs, action, reward, terminated, next_obs):
        """
        Updates the Q-table using the Bellman equation.
        
        Update process:
        1. Calculates the best possible future value
        2. Calculates the TD error (difference between expectation and reality)
        3. Adjusts the Q value in proportion to the error
        
        Numeric example:
        ---------------
        Situation:
          - Current state: s=2
          - Action taken: a=1 (down)
          - Reward: r=1
          - Next state: s'=3
          - Current Q(2,1) = 0.5
          - max_a' Q(3,a') = 0.8
          - lr = 0.1, gamma = 0.9
        
        Calculations:
          - future_q_value = 0.8
          - td_error = 1 + 0.9x0.8 - 0.5 = 1.22
          - New Q(2,1) = 0.5 + 0.1x1.22 = 0.622
        
        Result:
          The cell [2][1] increases from 0.5 -> 0.622
          The agent learned that action is better than thought
        
        Args:
            obs (int): Current state
            action (int): Action taken
            reward (float): Reward received
            terminated (bool): If the episode ended
            next_obs (int): Next state
        """
        # ===== FUTURE VALUE CALCULATION =====
        # If the episode ended: future_q_value = 0 (no future)
        # If it continues: future_q_value = max_a' Q(s', a') (best possible value in next state)
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        
        # ===== STEP 2: CALCULATE TEMPORAL DIFFERENCE (TD) ERROR =====
        # Measures how wrong we were in our prediction
        # TD error = (Actual reward + Future value) - Previous prediction
        td_error = reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        
        # Interpretation of the TD error:
        # - TD > 0: The action was BETTER than expected -> increases Q
        # - TD < 0: The action was WORSE than expected -> decreases Q
        # - TD ~= 0: The prediction was CORRECT -> minimal change
        
        # ===== STEP 3: UPDATE Q-TABLE =====
        # Only modify the corresponding Q(s,a) cell
        # Update proportional to the error: larger error -> larger change
        self.q_values[obs][action] += self.lr * td_error
        
        # ===== STEP 4: RECORD ERROR FOR ANALYSIS =====
        # Save the TD error to monitor learning
        # A decreasing error indicates the agent is converging
        self.training_error.append(td_error)

    def decay_epsilon(self):
        """
        Linearly reduces epsilon to decrease exploration.
        
        Executed at the end of each episode. As the agent learns,
        it explores less and exploits its knowledge more.
        
        Typical evolution example:
          Episode 0:     epsilon = 1.0  (100% exploration)
          Episode 5000:  epsilon = 0.55 (55% exploration)
          Episode 10000: epsilon = 0.1  (10% exploration, holds steady)
        """
        # Reduce epsilon but never below the minimum
        # Keeps a residual exploration chance to discover changes
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

# ========================================
# TRAINING FUNCTION
# ========================================

def train_volcanoLake_agent(n_episodes):
    """
    Trains a Q-Learning agent in the FrozenLake environment.
    
    Environment configuration:
    - FrozenLake-v1: 4x4 Grid with slippery tiles (is_slippery=True)
    - Slippery tiles simulate slippery ice (realism)
    - Makes learning harder but more robust
    
    Args:
        n_episodes (int): Number of training episodes
        
    Returns:
        tuple: (env, agent) - Environment and trained agent
    """
    # ===== ENVIRONMENT CONFIGURATION =====
    # Creates default FrozenLake environment
    env = gym.make("FrozenLake-v1", is_slippery = True)
    
    # Wrapper to record statistics per episode
    # Saves accumulated rewards and episode durations
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    # ===== HYPERPARAMETER CONFIGURATION =====
    learning_rate = 0.01
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.1

    # ===== AGENT INITIALIZATION =====
    agent = VolcanoLakeAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    # ===== TRAINING LOOP =====
    # Iterates through n_episodes with progress bar
    for _ in tqdm(range(n_episodes)):
        # Reset the environment after each episode
        obs, _ = env.reset() 
        done = False
        # Each iteration is an agent step
        while not done:
            # Choose action according to defined policy (epsilon-greedy)
            action = agent.get_action(env, obs) 
            # Executes the action "filling" the variables
            next_obs, reward, terminated, truncated, _ = env.step(action) 
            # Updates the Q-table
            agent.update(obs, action, reward, terminated, next_obs) 
            # Update current state
            obs = next_obs 
            # End of episode
            done = terminated or truncated 
        # Reduce epsilon after each episode
        agent.decay_epsilon() 

    return env, agent

# ========================================
# MAIN EXECUTION BLOCK
# ========================================

if __name__ == "__main__":
    # ===== CONFIGURATION =====
    plot_save = True
    n_episodes = 100_000
    
    # ===== TRAINING =====
    env, agent = train_volcanoLake_agent(n_episodes)
    
    # ===== RESULTS VISUALIZATION =====
    plot_training(env, agent, plot_save)
    
    # ===== Q-TABLE ANALYSIS =====
    # Configuration to print arrays legibly
    np.set_printoptions(precision=2, suppress=True)
    print("\nQ-table final (filas=estados, columnas=acciones):\n")
    print(agent.q_values)
    
    print("\nMejor acción por estado (0=izq, 1=abajo, 2=dcha, 3=arriba):")
    best_actions = np.argmax(agent.q_values, axis=1).reshape(4, 4)    
    print(best_actions)
    
    # Close the environment
    env.close()