import gymnasium as gym
from tqdm import tqdm
import os

from envs.volcano_lake_env import VolcanoLakeEnv
from agent.qlearning_agent import VolcanoLakeAgent    
from wrappers.wrappers import IncreasingLavaHoles, LimitedVision, ActionFlickerWrapper, TorusWrapper    

# Get project directory (VolcanoLake_v3)
current_dir = os.path.dirname(os.path.abspath(__file__)) # training/
project_root = os.path.dirname(current_dir) # VolcanoLake_v3/

# Registration of the custom environment in gymnasium
gym.register(
    id='VolcanoLake-v3',
    entry_point='envs.volcano_lake_env:VolcanoLakeEnv',
    kwargs={'map_file_path': 'VolcanoLake_v3/maps/map_25x25.csv'} 
)

def train_volcanoLake_agent(n_episodes, map_path, learning_rate, start_epsilon, epsilon_decay, final_epsilon):
    """
    Trains a Q-Learning agent in the VolcanoLake environment.
    
    Args:
        n_episodes (int): Number of training episodes
        map_path (str): Path to the environment map file
        learning_rate (float): Agent learning rate
        start_epsilon (float): Initial epsilon value for exploration
        epsilon_decay (float): Epsilon decay factor
        final_epsilon (float): Minimum final epsilon value
    
    Returns:
        tuple: (environment, agent) trained
    """
    # Create videos folder inside VolcanoLake_v3
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


    # Agent initialization
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
            # obs is an int
            action = agent.get_action(env, obs) 
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.update(obs, action, reward, terminated, next_obs)
            obs = next_obs
            done = terminated or truncated
        agent.decay_epsilon()

    return env, agent