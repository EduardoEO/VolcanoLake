# Authors: Eduardo Estefanía Ovejero & Álvaro Martín García 

import gymnasium as gym
import numpy as np
from tqdm import tqdm
import os
import warnings

from utils import plot_training
from agent import VolcanoLakeAgent
from wrappers import IncreasingHoles, LimitedVisionRewardShaping, LimitedVision

warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.wrappers.rendering")


def train_volcanoLake_agent(n_episodes):
    """
    Main function to train a Q-Learning agent in the VolcanoLake environment.
    
    Configures the environment with multiple wrappers, initializes the agent, and runs
    the training loop for the specified number of episodes.
    
    Args:
        n_episodes (int): Total number of episodes for training
        
    Returns:
        tuple: (env, agent) - Configured environment and trained agent
    """
    # ===== ENVIRONMENT CONFIGURATION =====
    # IMPORTANT: The order of wrappers is critical
    # Wrappers that modify environment mechanics go first
    # Observation and statistics wrappers go last
    
    # Base environment: FrozenLake without slippage
    env = gym.make("FrozenLake-v1", is_slippery = False, render_mode = "rgb_array")
    
    # --- Wrappers ---
    env = IncreasingHoles(env, 2, 0.0001)
    # Previous version: Only saw the right cell
    # env = LimitedVisionRewardShaping(env)
    # Current version: Will see the cell in front according to previous action
    env = LimitedVision(env)
    env = gym.wrappers.TimeLimit(env, 20)
    
    # Create videos folder in the script directory
    video_folder = os.path.join(os.path.dirname(__file__), "videos")
    os.makedirs(video_folder, exist_ok=True)
    env = gym.wrappers.RecordVideo(
        env, 
        video_folder = video_folder,
        episode_trigger = lambda ep: ep == n_episodes - 1
    )
    
    # Statistics recording (MUST BE AT THE END)
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
    for _ in tqdm(range(n_episodes)):
        # Reset the environment after each episode
        obs, info = env.reset()
        done = False
        # Each iteration equals one agent step
        while not done:
            # Choose action according to defined policy (epsilon-greedy)
            action = agent.get_action(env, obs) 
            # Execute the action "filling" the variables
            next_obs, reward, terminated, truncated, info = env.step(action) 
            # Update the Q-table
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
    # ===== EXECUTION CONFIGURATION =====
    plot_save = True
    n_episodes = 100_000
    
    # ===== AGENT TRAINING =====
    env, agent = train_volcanoLake_agent(n_episodes)
    
    # ===== GENERATING TRAINING PLOTS =====
    plot_training(env, agent, plot_save)
    
    # ===== RESULTS ANALYSIS =====
    # Configuration to display arrays more legibly
    np.set_printoptions(precision=2, suppress=True)
    print("\nQ-table final (filas=estados, columnas=acciones):")
    print(agent.q_values)
    
    print("\nMejor acción por estado (0=izq, 1=abajo, 2=dcha, 3=arriba):")
    best_actions = np.argmax(agent.q_values, axis=1).reshape(4, 4)    
    print(best_actions)
    
    print("\nMapa final:")
    # Decode bytes to strings for clean display
    decoded_map = [[cell.decode('utf-8') for cell in row] for row in env.unwrapped.desc]
    for row in decoded_map:
        print(' '.join(row))
    
    # Close the environment
    env.close()