import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# Función para visualización
# -----------------------------

def plot_training(env, agent, plot_save=False, rolling_length=500):
    fig, axs = plt.subplots(ncols=4, figsize=(15, 5))
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
    
    # Tasa de éxito
    # Asumimos que un "éxito" es una recompensa final > 5
    win_episodes = (np.array(env.return_queue).flatten() > 5).astype(float)
    win_rate_ma = np.convolve(win_episodes, 
                          np.ones(rolling_length), mode="valid") / rolling_length
    axs[3].plot(range(len(win_rate_ma)), win_rate_ma)
    axs[3].set_title("Tasa de Éxito (Meta alcanzada)")
    axs[3].set_ylim(0, 1.05) # Fija el eje Y entre 0 y 100%

    plt.tight_layout()
    
    if plot_save:
        # Obtener directorio del proyecto (VolcanoLake_v3)
        script_dir = os.path.dirname(os.path.abspath(__file__)) # utils/
        project_root = os.path.dirname(script_dir) # VolcanoLake_v3/
        plots_dir = os.path.join(project_root, "plots") # VolcanoLake_v3/plots/
        
        # Crear directorio si no existe
        os.makedirs(plots_dir, exist_ok=True)
        
        # Guardar con ruta absoluta
        save_path = os.path.join(plots_dir, "volcanolake_training_metrics.png")
        
        plt.savefig(save_path)
    else:
        plt.show()