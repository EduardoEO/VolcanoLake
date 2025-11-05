import gymnasium as gym
import numpy as np
from tqdm import tqdm
from utils import plot_training
from agent import VolcanoLakeAgent
from wrappers import IncreasingHoles, LimitedVisionRewardShaping, LimitedVision

def train_volcanoLake_agent(n_episodes):
    # El orden de los wrappers importa mucho. Primero deben ir los wrappers que más cambien el entorno más cerca del env original y al final los de recolección de estdísticas u de observación. Por ejemplo, al poner primero el de stats luego no te deja usar env.return_queue o env.length_queue.
    env = gym.make("FrozenLake-v1", is_slippery = False, render_mode = "rgb_array") # Inicializa el entorno con casillas deslizantes. Esto hace que le cueste más al agente aprender una ruta óptima.
    env = IncreasingHoles(env, 2, 0.0001)
    #env = LimitedVisionRewardShaping(env) # Verá solo la casilla de la derecha
    env = LimitedVision(env) # Verá la casilla de enfrente según la acción anterior
    env = gym.wrappers.TimeLimit(env, 20)
    env = gym.wrappers.RecordVideo(
        env, 
        video_folder = "videos/",
        episode_trigger = lambda ep: ep == n_episodes - 1
    )
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
    n_episodes = 500_000
    
    env, agent = train_volcanoLake_agent(n_episodes)
    plot_training(env, agent, plot_save)
    
    np.set_printoptions(precision=2, suppress=True)
    print("\nQ-table final (filas=estados, columnas=acciones):")
    print(agent.q_values)
    
    print("\nMejor acción por estado (0=izq, 1=abajo, 2=dcha, 3=arriba):")
    best_actions = np.argmax(agent.q_values, axis=1).reshape(4, 4)    
    print(best_actions)
    
    print("\nMapa final:")
    print(env.unwrapped.desc)
    
    env.close() # Close the environment