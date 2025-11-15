import gymnasium as gym
from tqdm import tqdm
import os

from envs.volcano_lake_env import VolcanoLakeEnv
from agent.qlearning_agent import VolcanoLakeAgent    
from wrappers.wrappers import IncreasingLavaHoles, LimitedVision, ActionFlickerWrapper, TorusWrapper    

# Obtener directorio del proyecto (VolcanoLake_v3)
current_dir = os.path.dirname(os.path.abspath(__file__)) # training/
project_root = os.path.dirname(current_dir) # VolcanoLake_v3/

# Registro del entorno personalizado en gymnasium
gym.register(
    id='VolcanoLake-v3',
    entry_point='envs.volcano_lake_env:VolcanoLakeEnv',
    kwargs={'map_file_path': 'VolcanoLake_v3/maps/map_25x25.csv'} 
)

def train_volcanoLake_agent(n_episodes, map_path, learning_rate, start_epsilon, epsilon_decay, final_epsilon):
    """
    Entrena un agente de Q-Learning en el entorno VolcanoLake.
    
    Args:
        n_episodes (int): Número de episodios de entrenamiento
        map_path (str): Ruta al archivo del mapa del entorno
        learning_rate (float): Tasa de aprendizaje del agente
        start_epsilon (float): Valor inicial de epsilon para exploración
        epsilon_decay (float): Factor de decaimiento de epsilon
        final_epsilon (float): Valor final mínimo de epsilon
    
    Returns:
        tuple: (entorno, agente) entrenados
    """
    # Crear carpeta de videos dentro de VolcanoLake_v3
    videos_dir = os.path.join(project_root, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    
    env = VolcanoLakeEnv(map_file_path=map_path, render_mode="rgb_array")
        
    # ---- Wrappers ----
    
    env = IncreasingLavaHoles(env, 5, 0.05)
    env = LimitedVision(env)
    env = TorusWrapper(env)
    env = ActionFlickerWrapper(env)
    env = gym.wrappers.TimeLimit(env, 100)
    
    env = gym.wrappers.RecordVideo(
        env, 
        video_folder = videos_dir,
        episode_trigger = lambda ep: ep == n_episodes - 1
    )
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)


    # Inicialización del agente
    agent = VolcanoLakeAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    for _ in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False
        while not done:
            action = agent.get_action(env, obs) # obs es un int
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.update(obs, action, reward, terminated, next_obs)
            obs = next_obs
            done = terminated or truncated
        agent.decay_epsilon()

    return env, agent