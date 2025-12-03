import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# Visualization functions
# -----------------------------

def plot_training(env, agent, plot_save=False, rolling_length=500):
    """
    Generates training metric plots for the Q-Learning agent.
    
    Creates a set of 5 visualizations distributed in a 2x3 layout:
    - Accumulated rewards per episode (smoothed)
    - Episode duration (steps per episode)  
    - TD (Temporal Difference) training error
    - Success rate (percentage of episodes reaching the goal)
    - Treasures collected per episode (if available)
    
    Args:
        env: Training environment (with RecordEpisodeStatistics wrappers)
        agent: Trained Q-Learning agent with error history
        plot_save (bool): If True, saves plots to plots/ folder
        rolling_length (int): Window for curve smoothing (moving average)
    
    Returns:
        None: Shows or saves plots according to plot_save
    """
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))
    fig.suptitle("Entrenamiento del agente volcanoLake", fontsize=16)

    # Rewards
    reward_ma = np.convolve(np.array(env.return_queue).flatten(),
                            np.ones(rolling_length), mode="valid") / rolling_length
    axs[0,0].plot(range(len(reward_ma)), reward_ma)
    axs[0,0].set_title("Recompensas por episodio")

    # Episode duration
    length_ma = np.convolve(np.array(env.length_queue).flatten(),
                            np.ones(rolling_length), mode="same") / rolling_length
    axs[0,1].plot(range(len(length_ma)), length_ma)
    axs[0,1].set_title("Duración de episodios")

    # Training error
    td_ma = np.convolve(np.array(agent.training_error),
                        np.ones(rolling_length), mode="same") / rolling_length
    axs[0,2].plot(range(len(td_ma)), td_ma)
    axs[0,2].set_title("Error de entrenamiento (TD)")
    
    # Success rate
    win_episodes = np.array(env.unwrapped.success_queue).flatten()
    win_rate_ma = np.convolve(win_episodes, 
                              np.ones(rolling_length), mode="valid") / rolling_length
    axs[1,0].plot(range(len(win_rate_ma)), win_rate_ma)
    axs[1,0].set_title("Tasa de Éxito (Meta alcanzada)")
    # Fix the Y axis between 0 and 100%
    axs[1,0].set_ylim(0, 1.05) 
    
    # Average treasures collected per episode (access original environment)
    try:
        # Try to access the attribute in the unwrapped environment
        treasures_data = np.array(env.unwrapped.treasures_found_queue).flatten()
        treasure_ma = np.convolve(treasures_data,
                                  np.ones(rolling_length), mode="valid") / rolling_length
        axs[1,1].plot(range(len(treasure_ma)), treasure_ma)
        axs[1,1].set_title("Tesoros recogidos por episodio")
    except (AttributeError, ValueError):
        # If the attribute does not exist or is empty, show message
        axs[1,1].text(0.5, 0.5, 'No hay datos de tesoros disponibles', 
                      horizontalalignment='center', verticalalignment='center',
                      transform=axs[1,1].transAxes)
        axs[1,1].set_title("Tesoros recogidos por episodio")

    # Hide the empty subplot
    axs[1,2].axis('off')

    plt.tight_layout()
    
    if plot_save:
        # Get project directory (VolcanoLake_v3)
        script_dir = os.path.dirname(os.path.abspath(__file__)) # utils/
        project_root = os.path.dirname(script_dir) # VolcanoLake_v3/
        plots_dir = os.path.join(project_root, "plots") # VolcanoLake_v3/plots/
        os.makedirs(plots_dir, exist_ok=True) # Create directory if it does not exist

        
        # Save with absolute path
        save_path = os.path.join(plots_dir, "volcanolake_training_metrics.png")
        
        plt.savefig(save_path)
    else:
        plt.show()
            
def plot_value_heatmap(env, agent, plot_dir="plots"):
    """
    Generates and saves a heatmap of the Q-Table values of the trained agent.
    
    Visualizes the maximum values of each state (max Q-value) as a heatmap
    over the environment grid, allowing identification of which states are most valuable
    to the agent after training.
    
    Args:
        env: VolcanoLake environment (with access to map dimensions)
        agent: Trained Q-Learning agent (with complete Q-table)
        plot_dir (str): Directory where to save the heatmap PNG file
    
    Returns:
        None: Saves the heatmap as a PNG file
    """
    os.makedirs(plot_dir, exist_ok=True)

    best_values = np.max(agent.q_values, axis=1)
    value_heatmap = best_values.reshape((env.unwrapped.nrows, env.unwrapped.ncols))

    plt.figure(figsize=(10, 10))
    plt.imshow(value_heatmap, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Valor del Estado (V(s))')
    plt.title(f"Heatmap de Valores (Q-Table) - Mapa {env.unwrapped.nrows}x{env.unwrapped.ncols}")

    heatmap_path = os.path.join(plot_dir, "volcanolake_value_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()

def plot_policy_map(env, agent, plot_dir="plots"):
    """
    Generates and saves a map of the trained agent's optimal policy.
    
    Visualizes the optimal actions of each state as directional arrows
    on a grid representing the environment. The arrows indicate the direction
    of movement the agent considers best according to its trained Q-table.
    No arrows are shown in terminal states (Lava 'L' and Goal 'G').
    
    Args:
        env: VolcanoLake environment (with access to map dimensions and description)
        agent: Trained Q-Learning agent (with complete Q-table)
        plot_dir (str): Directory where to save the policy map PNG file
    
    Returns:
        None: Saves the policy map as a PNG file
    """
    os.makedirs(plot_dir, exist_ok=True)
    unwrapped_env = env.unwrapped

    fig, ax = plt.subplots(figsize=(12, 10))

    best_actions_map = np.argmax(agent.q_values, axis=1).reshape((unwrapped_env.nrows, unwrapped_env.ncols))

    action_to_uv = {
        action: (delta[1], delta[0]) 
        for action, delta in unwrapped_env.action_to_delta.items()
    }
    
    U = np.zeros_like(best_actions_map, dtype=float)
    V = np.zeros_like(best_actions_map, dtype=float)
    
    for r in range(unwrapped_env.nrows):
        for c in range(unwrapped_env.ncols):
            action = best_actions_map[r, c]
            u, v = action_to_uv[action]
            
            # Do not draw arrows in terminal tiles (Lava, Goal)
            tile = unwrapped_env.base_desc[r, c]
            if tile in ['G', 'L']:
                U[r, c] = 0
                V[r, c] = 0
            else:
                U[r, c] = u
                V[r, c] = v

    X, Y = np.meshgrid(np.arange(unwrapped_env.ncols), np.arange(unwrapped_env.nrows))
    
    ax.quiver(X, Y, U, V, color='black', pivot='middle', headwidth=4, headlength=5)
    
    # We invert the Y axis so that (0,0) is at the top left
    ax.invert_yaxis() 
    
    ax.set_title(f"Mapa de Política (Q-Table) - Mapa {unwrapped_env.nrows}x{unwrapped_env.ncols}")
    ax.set_xticks(np.arange(unwrapped_env.ncols))
    ax.set_yticks(np.arange(unwrapped_env.nrows))
    ax.set_xticklabels(np.arange(unwrapped_env.ncols))
    ax.set_yticklabels(np.arange(unwrapped_env.nrows))
    
    # We add a grid to make it easier to read
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_aspect('equal') # Makes the tiles square

    policy_path = os.path.join(plot_dir, "volcanolake_policy_map.png")
    plt.savefig(policy_path)
    plt.close(fig)