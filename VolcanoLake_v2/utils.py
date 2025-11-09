import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Función para visualización
# -----------------------------

def plot_training(env, agent, plot_save=False, rolling_length=500):
    fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
    fig.suptitle("Entrenamiento del agente volcanoLake", fontsize=16)

    # Recompensas
    reward_ma = np.convolve(np.array(env.return_queue).flatten(),
                            np.ones(rolling_length), mode="valid") / rolling_length
    axs[0].plot(range(len(reward_ma)), reward_ma)
    axs[0].set_title("Recompensas por episodio")

    # Duración de episodios
    length_ma = np.convolve(np.array(env.length_queue).flatten(),
                            np.ones(rolling_length), mode="same") / rolling_length
    axs[1].plot(range(len(length_ma)), length_ma)
    axs[1].set_title("Duración de episodios")

    # Error de entrenamiento
    td_ma = np.convolve(np.array(agent.training_error),
                        np.ones(rolling_length), mode="same") / rolling_length
    axs[2].plot(range(len(td_ma)), td_ma)
    axs[2].set_title("Error de entrenamiento (TD)")

    plt.tight_layout()
    if plot_save:
        plt.savefig("plots/volcanolake_training_metrics.png")
    else:
        plt.show()