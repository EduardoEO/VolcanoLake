import gymnasium as gym
import numpy as np
from tqdm import tqdm
from utils import plot_training
from agent import VolcanoLakeAgent
from wrappers import IncreasingHoles, LimitedVisionRewardShaping, LimitedVision

def train_volcanoLake_agent(n_episodes):
    """
    Función principal para entrenar un agente Q-Learning en el entorno VolcanoLake.
    
    Configura el entorno con múltiples wrappers, inicializa el agente y ejecuta
    el bucle de entrenamiento durante el número especificado de episodios.
    
    Args:
        n_episodes (int): Número total de episodios para el entrenamiento
        
    Returns:
        tuple: (env, agent) - Entorno configurado y agente entrenado
    """
    # ===== CONFIGURACIÓN DEL ENTORNO =====
    # IMPORTANTE: El orden de los wrappers es crítico
    # Los wrappers que modifican la mecánica del entorno van primero
    # Los wrappers de observación y estadísticas van al final
    
    # Entorno base: FrozenLake sin deslizamiento
    env = gym.make("FrozenLake-v1", is_slippery = False, render_mode = "rgb_array")
    
    # --- Wrappers ---
    env = IncreasingHoles(env, 2, 0.0001)
    #env = LimitedVisionRewardShaping(env) # Verá solo la casilla de la derecha
    env = LimitedVision(env) # Verá la casilla de enfrente según la acción anterior
    env = gym.wrappers.TimeLimit(env, 20)
    env = gym.wrappers.RecordVideo(
        env, 
        video_folder = "videos/",
        episode_trigger = lambda ep: ep == n_episodes - 1
    )
    # Registro de estadísticas (DEBE IR AL FINAL)
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
    for _ in tqdm(range(n_episodes)):
        # Reinicio del entrono después de cada episodio
        obs, info = env.reset()
        done = False
        # Un bucle del while equivale a un episodio
        while not done:
            action = agent.get_action(env, obs) # Elige la acción según la política definida (epsilon-greedy)
            next_obs, reward, terminated, truncated, info = env.step(action) # Ejecuta la acción "rellenando" las variables
            agent.update(obs, action, reward, terminated, next_obs) # Actualiza la Q-table
            obs = next_obs # Actualización del estado actual
            done = terminated or truncated # Fin del episodio
        agent.decay_epsilon() # Reducir el epsilon después de cada episodio

    return env, agent

# -----------------------------
# Bloque principal de ejecución
# -----------------------------

if __name__ == "__main__":
    # ===== CONFIGURACIÓN DE EJECUCIÓN =====
    plot_save = True
    n_episodes = 500_000
    
    # ===== ENTRENAMIENTO DEL AGENTE =====
    env, agent = train_volcanoLake_agent(n_episodes)
    
    # ===== GENERACIÓN DE GRÁFICAS DE ENTRENAMIENTO =====
    plot_training(env, agent, plot_save)
    
    # ===== ANÁLISIS DE RESULTADOS =====
    # Configuración para mostrar arrays de forma más legible
    np.set_printoptions(precision=2, suppress=True)
    print("\nQ-table final (filas=estados, columnas=acciones):")
    print(agent.q_values)
    
    print("\nMejor acción por estado (0=izq, 1=abajo, 2=dcha, 3=arriba):")
    best_actions = np.argmax(agent.q_values, axis=1).reshape(4, 4)    
    print(best_actions)
    
    print("\nMapa final:")
    print(env.unwrapped.desc)
    
    env.close() # Cerrar el entorno