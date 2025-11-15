import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# Función para visualización
# -----------------------------

def plot_training(env, agent, plot_save=False, rolling_length=500):
    """
    Genera gráficos de métricas del entrenamiento del agente Q-Learning.
    
    Crea un conjunto de 5 visualizaciones distribuidas en layout 2x3:
    - Recompensas acumuladas por episodio (suavizadas)
    - Duración de episodios (pasos por episodio)  
    - Error de entrenamiento TD (Temporal Difference)
    - Tasa de éxito (porcentaje de episodios que alcanzan la meta)
    - Tesoros recogidos por episodio (si están disponibles)
    
    Args:
        env: Entorno de entrenamiento (con wrappers RecordEpisodeStatistics)
        agent: Agente Q-Learning entrenado con historial de errores
        plot_save (bool): Si True, guarda las gráficas en carpeta plots/
        rolling_length (int): Ventana para suavizado de curvas (media móvil)
    
    Returns:
        None: Muestra o guarda las gráficas según plot_save
    """
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))
    fig.suptitle("Entrenamiento del agente volcanoLake", fontsize=16)

    # Recompensas
    reward_ma = np.convolve(np.array(env.return_queue).flatten(),
                            np.ones(rolling_length), mode="valid") / rolling_length
    axs[0,0].plot(range(len(reward_ma)), reward_ma)
    axs[0,0].set_title("Recompensas por episodio")

    # Duración de episodios
    length_ma = np.convolve(np.array(env.length_queue).flatten(),
                            np.ones(rolling_length), mode="same") / rolling_length
    axs[0,1].plot(range(len(length_ma)), length_ma)
    axs[0,1].set_title("Duración de episodios")

    # Error de entrenamiento
    td_ma = np.convolve(np.array(agent.training_error),
                        np.ones(rolling_length), mode="same") / rolling_length
    axs[0,2].plot(range(len(td_ma)), td_ma)
    axs[0,2].set_title("Error de entrenamiento (TD)")
    
    # Tasa de éxito
    win_episodes = np.array(env.unwrapped.success_queue).flatten()
    win_rate_ma = np.convolve(win_episodes, 
                          np.ones(rolling_length), mode="valid") / rolling_length
    axs[1,0].plot(range(len(win_rate_ma)), win_rate_ma)
    axs[1,0].set_title("Tasa de Éxito (Meta alcanzada)")
    axs[1,0].set_ylim(0, 1.05) # Fija el eje Y entre 0 y 100%
    
    # Media de tesoros recogidos por episodio (acceder al entorno original)
    try:
        # Intentar acceder al atributo en el entorno unwrapped
        treasures_data = np.array(env.unwrapped.treasures_found_queue).flatten()
        treasure_ma = np.convolve(treasures_data,
                                  np.ones(rolling_length), mode="valid") / rolling_length
        axs[1,1].plot(range(len(treasure_ma)), treasure_ma)
        axs[1,1].set_title("Tesoros recogidos por episodio")
    except (AttributeError, ValueError):
        # Si no existe el atributo o está vacío, mostrar mensaje
        axs[1,1].text(0.5, 0.5, 'No hay datos de tesoros disponibles', 
                      horizontalalignment='center', verticalalignment='center',
                      transform=axs[1,1].transAxes)
        axs[1,1].set_title("Tesoros recogidos por episodio")

    # Ocultar el subplot vacío
    axs[1,2].axis('off')

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
            
def plot_value_heatmap(env, agent, plot_dir="plots"):
    """
    Genera y guarda un heatmap del valor de la Q-Table del agente entrenado.
    
    Visualiza los valores máximos de cada estado (max Q-value) como un mapa de calor
    sobre la grilla del entorno, permitiendo identificar qué estados son más valiosos
    para el agente después del entrenamiento.
    
    Args:
        env: Entorno VolcanoLake (con acceso a dimensiones del mapa)
        agent: Agente Q-Learning entrenado (con Q-table completa)
        plot_dir (str): Directorio donde guardar el archivo PNG del heatmap
    
    Returns:
        None: Guarda el heatmap como archivo PNG
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
    Genera y guarda un mapa de la política óptima del agente entrenado.
    
    Visualiza las acciones óptimas de cada estado como flechas direccionales
    sobre una grilla que representa el entorno. Las flechas indican la dirección
    de movimiento que el agente considera mejor según su Q-table entrenada.
    No se muestran flechas en estados terminales (Lava 'L' y Meta 'G').
    
    Args:
        env: Entorno VolcanoLake (con acceso a dimensiones y descripción del mapa)
        agent: Agente Q-Learning entrenado (con Q-table completa)
        plot_dir (str): Directorio donde guardar el archivo PNG del mapa de política
    
    Returns:
        None: Guarda el mapa de política como archivo PNG
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
            
            # No dibujar flechas en casillas terminales (Lava, Meta)
            tile = unwrapped_env.base_desc[r, c]
            if tile in ['G', 'L']:
                U[r, c] = 0
                V[r, c] = 0
            else:
                U[r, c] = u
                V[r, c] = v

    X, Y = np.meshgrid(np.arange(unwrapped_env.ncols), np.arange(unwrapped_env.nrows))
    
    ax.quiver(X, Y, U, V, color='black', pivot='middle', headwidth=4, headlength=5)
    
    # Invertimos el eje Y para que (0,0) esté arriba a la izquierda
    ax.invert_yaxis() 
    
    ax.set_title(f"Mapa de Política (Q-Table) - Mapa {unwrapped_env.nrows}x{unwrapped_env.ncols}")
    ax.set_xticks(np.arange(unwrapped_env.ncols))
    ax.set_yticks(np.arange(unwrapped_env.nrows))
    ax.set_xticklabels(np.arange(unwrapped_env.ncols))
    ax.set_yticklabels(np.arange(unwrapped_env.nrows))
    
    # Añadimos una rejilla para que sea más fácil de leer
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_aspect('equal') # Hace que las casillas sean cuadradas

    policy_path = os.path.join(plot_dir, "volcanolake_policy_map.png")
    plt.savefig(policy_path)
    plt.close(fig)