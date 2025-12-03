import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# Visualization functions
# -----------------------------

def plot_training(env, agent, plot_save=False, rolling_length=500):
    """
    Generates training metric plots for the Q-Learning agent.
    
    Creates a set of 3 visualizations distributed in a 1x3 layout:
    - Accumulated rewards per episode (smoothed with moving average)
    - Episode duration (number of steps per episode)  
    - TD (Temporal Difference) training error
    
    Args:
        env: Training environment (with RecordEpisodeStatistics wrapper)
             Must have return_queue and length_queue attributes with history
        agent: Q-Learning agent trained with error history
               Must have training_error attribute with list of TD errors
        plot_save (bool): If True, saves plots to plots/ folder
                         If False, shows plots on screen
        rolling_length (int): Window for curve smoothing (moving average)
                             Default value: 500 episodes
    
    Returns:
        None: Shows or saves plots according to plot_save parameter
    """
    fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
    fig.suptitle("Entrenamiento del agente volcanoLake", fontsize=16)

    # Rewards
    reward_ma = np.convolve(np.array(env.return_queue).flatten(),
                            np.ones(rolling_length), mode="valid") / rolling_length
    axs[0].plot(range(len(reward_ma)), reward_ma)
    axs[0].set_title("Recompensas por episodio")

    # Episode durations
    length_ma = np.convolve(np.array(env.length_queue).flatten(),
                            np.ones(rolling_length), mode="same") / rolling_length
    axs[1].plot(range(len(length_ma)), length_ma)
    axs[1].set_title("Duración de episodios")

    # Training error
    td_ma = np.convolve(np.array(agent.training_error),
                        np.ones(rolling_length), mode="same") / rolling_length
    axs[2].plot(range(len(td_ma)), td_ma)
    axs[2].set_title("Error de entrenamiento (TD)")

    # Adjust spacing between subplots
    plt.tight_layout()
    
    # Save or show according to the plot_save parameter
    if plot_save:
        # Create plots folder on current directory
        plots_folder = os.path.join(os.path.dirname(__file__), "plots")
        os.makedirs(plots_folder, exist_ok=True)
        plt.savefig(os.path.join(plots_folder, "volcanolake_training_metrics.png"))
    else:
        plt.show()