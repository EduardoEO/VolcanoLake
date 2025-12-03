# Autores: Eduardo Estefanía Ovejero y Álvaro Martín García

import numpy as np
import warnings

from training.train_agent import train_volcanoLake_agent
from utils.plotting import plot_training, plot_value_heatmap, plot_policy_map

warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.wrappers.rendering")

if __name__ == "__main__":
    
    # ==================== PARAMETER CONFIGURATION ====================

    # Training parameters
    N_EPISODES = 100_000
    MAP_FILE = "VolcanoLake_v3/maps/map_25x25.csv"
    
    # Q-Learning algorithm parameters
    LEARNING_RATE = 0.1
    START_EPSILON = 1.0
    EPSILON_DECAY = START_EPSILON / (N_EPISODES / 2) 
    FINAL_EPSILON = 0.1
    
    # ==================== TRAINING EXECUTION ====================
    
    # Train the agent with the configured parameters
    env, agent = train_volcanoLake_agent(
        n_episodes=N_EPISODES,
        map_path=MAP_FILE,
        learning_rate=LEARNING_RATE,
        start_epsilon=START_EPSILON,
        epsilon_decay=EPSILON_DECAY,
        final_epsilon=FINAL_EPSILON
    )
    
    # ==================== RESULTS ANALYSIS ====================
    
    # Generate and save plots
    plot_save = True 
    plot_training(env, agent, plot_save)
    plot_value_heatmap(env, agent, plot_dir="VolcanoLake_v3/plots")
    plot_policy_map(env, agent, plot_dir="VolcanoLake_v3/plots")
    
    # Show a sample of the final Q-table with 2 decimals and without scientific notation
    np.set_printoptions(precision=2, suppress=True)
    print("\nQ-table final (primeros 25 estados):")
    print(agent.q_values[:25])
    
    # Show the learned optimal policy (best action per state) of the final Q-table
    print("\nMejor acción por estado (0-Arriba, 1-Der, 2-Aba, 3-Izq, 4-ArrDer, 5-AbaIzq, 6-AbaIzq, 7-ArrIzq):")
    best_actions = np.argmax(agent.q_values, axis=1).reshape(env.unwrapped.nrows, env.unwrapped.ncols)    
    print(best_actions)
    
    # Show the final map state after the last episode
    print("\nMapa final (con tesoros consumidos si el último episodio los cogió):")
    print(env.unwrapped.current_desc)
    
    env.close()