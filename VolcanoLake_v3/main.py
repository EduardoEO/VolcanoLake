import numpy as np
import warnings

from training.train_agent import train_volcanoLake_agent
from utils.plotting import plot_training, plot_value_heatmap

warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.wrappers.rendering")

if __name__ == "__main__":
    
    # ==================== CONFIGURACIÓN DE PARÁMETROS ====================

    # Parametros del entrenamiento
    N_EPISODES = 100_000
    MAP_FILE = "VolcanoLake_v3/maps/map_5x5.csv"
    
    #Parámetros del algoritmo Q-Learning
    LEARNING_RATE = 0.1
    START_EPSILON = 1.0
    EPSILON_DECAY = START_EPSILON / (N_EPISODES / 2) 
    FINAL_EPSILON = 0.1
    
    # ==================== EJECUCIÓN DEL ENTRENAMIENTO ====================
    
    # Entrenar al agente con los parámetros configurados
    env, agent = train_volcanoLake_agent(
        n_episodes=N_EPISODES,
        map_path=MAP_FILE,
        learning_rate=LEARNING_RATE,
        start_epsilon=START_EPSILON,
        epsilon_decay=EPSILON_DECAY,
        final_epsilon=FINAL_EPSILON
    )
    
    # ==================== ANÁLISIS DE RESULTADOS ====================
    
    # Generar y guardar gráficos
    plot_save = True 
    plot_training(env, agent, plot_save)
    plot_value_heatmap(env, agent, plot_dir="VolcanoLake_v3/plots")
    
    # Mostrar una muestra de la Q-table final con 2 decimales y sin notación científica
    np.set_printoptions(precision=2, suppress=True)
    print("\nQ-table final (primeros 25 estados):")
    print(agent.q_values[:25])
    
    # Mostrar la política óptima aprendida (mejor acción por cada estado) de la Q-table final
    print("\nMejor acción por estado (0-Arriba, 1-Der, 2-Aba, 3-Izq, 4-ArrDer, 5-AbaIzq, 6-AbaIzq, 7-ArrIzq):")
    best_actions = np.argmax(agent.q_values, axis=1).reshape(env.unwrapped.nrows, env.unwrapped.ncols)    
    print(best_actions)
    
    # Mostrar el estado final del mapa después del último episodio
    print("\nMapa final (con tesoros consumidos si el último episodio los cogió):")
    print(env.unwrapped.current_desc)
    
    env.close()