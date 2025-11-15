import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# Función para visualización
# -----------------------------

def plot_training(env, agent, plot_save=False, rolling_length=500):
    """
    Genera gráficos de métricas del entrenamiento del agente Q-Learning.
    
    Crea un conjunto de 3 visualizaciones distribuidas en layout 1x3:
    - Recompensas acumuladas por episodio (suavizadas con media móvil)
    - Duración de episodios (número de pasos por episodio)  
    - Error de entrenamiento TD (Temporal Difference Error)
    
    Args:
        env: Entorno de entrenamiento (con wrapper RecordEpisodeStatistics)
             Debe tener atributos return_queue y length_queue con historial
        agent: Agente Q-Learning entrenado con historial de errores
               Debe tener atributo training_error con lista de errores TD
        plot_save (bool): Si True, guarda las gráficas en carpeta plots/
                         Si False, muestra las gráficas en pantalla
        rolling_length (int): Ventana para suavizado de curvas (media móvil)
                             Valor por defecto: 500 episodios
    
    Returns:
        None: Muestra o guarda las gráficas según parámetro plot_save
    """
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
        # Crear carpeta plots en el directorio del script
        plots_folder = os.path.join(os.path.dirname(__file__), "plots")
        os.makedirs(plots_folder, exist_ok=True)
        plt.savefig(os.path.join(plots_folder, "volcanolake_training_metrics.png"))
    else:
        plt.show()