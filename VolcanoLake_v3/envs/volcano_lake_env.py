import gymnasium as gym
import numpy as np
import os

class VolcanoLakeEnv(gym.Env):
    """
    Entorno personalizado VolcanoLake v3.

    Reglas:
    - S: Inicio
    - G: Meta (+10, termina)
    - L: Lava (-10, termina)
    - W: Agua (-1, resbala)
    - T: Tesoro (+5, continúa, se consume)
    - .: Tierra (0)
    
    Acciones (8):
    - 0: Arriba
    - 1: Derecha
    - 2: Abajo
    - 3: Izquierda
    - 4: Arriba-Derecha
    - 5: Abajo-Derecha
    - 6: Abajo-Izquierda
    - 7: Arriba-Izquierda
    
    Lógica de Agua (W):
    - 80% de éxito.
    - 10% de resbalar a 90° izq (cardinal) o componente 1 (diagonal).
    - 10% de resbalar a 90° der (cardinal) o componente 2 (diagonal).
    """
    
    # Opcional: metadata para el modo render
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, map_file_path):
        super().__init__()

        # --- Carga del mapa ---
        if not os.path.exists(map_file_path):
            raise FileNotFoundError(f"No se pudo encontrar el archivo del mapa: {map_file_path}")
            
        # Cargamos el mapa base (inmutable)
        # Usamos 'U' para strings (Unicode)
        self.base_desc = np.genfromtxt(map_file_path, dtype='<U1', delimiter=',')
        
        # Creamos una copia mutable para la sesión actual (para consumir tesoros)
        self.current_desc = np.copy(self.base_desc)
        
        self.nrows, self.ncols = self.base_desc.shape

        # --- Definición de Espacios ---
        # El espacio de observación es un único entero que representa la posición
        self.observation_space = gym.spaces.Discrete(self.nrows * self.ncols)
        
        # El espacio de acciones tiene 8 acciones
        self.action_space = gym.spaces.Discrete(8)

        # --- Mapeo de Acciones ---
        # (delta_fila, delta_columna)
        self.action_to_delta = {
            0: (-1, 0),  # Arriba
            1: (0, 1),   # Derecha
            2: (1, 0),   # Abajo
            3: (0, -1),  # Izquierda
            4: (-1, 1),  # Arriba-Derecha
            5: (1, 1),   # Abajo-Derecha
            6: (1, -1),  # Abajo-Izquierda
            7: (-1, -1)  # Arriba-Izquierda
        }
        
        # --- Lógica de deslizamiento (Agua 'W') ---
        # Para acciones cardinales (0-3), mapea a las acciones 90° izq/der
        self.slip_map_cardinal = {
            0: (3, 1),  # Arriba -> Izq, Der
            1: (0, 2),  # Derecha -> Arr, Aba
            2: (1, 3),  # Abajo -> Der, Izq
            3: (2, 0)   # Izquierda -> Aba, Arr
        }
        
        # Para acciones diagonales (4-7), mapea a los *deltas* de sus componentes
        self.slip_map_diagonal = {
            4: ((-1, 0), (0, 1)),  # Arriba-Der -> Arriba, Derecha
            5: ((1, 0), (0, 1)),   # Abajo-Der -> Abajo, Derecha
            6: ((1, 0), (0, -1)),  # Abajo-Izq -> Abajo, Izquierda
            7: ((-1, 0), (0, -1))  # Arriba-Izq -> Arriba, Izquierda
        }

        # --- Estado Inicial ---
        # Encontrar la 'S'
        start_pos_array = np.where(self.base_desc == 'S')
        if start_pos_array[0].size == 0:
            raise ValueError("El mapa no contiene un punto de inicio 'S'")
        
        self.start_pos = (start_pos_array[0][0], start_pos_array[1][0])
        self.start_state = self._pos_to_state(self.start_pos[0], self.start_pos[1])
        
        # El estado actual del agente (se reiniciará en reset())
        self.agent_state = self.start_state


    def _pos_to_state(self, row, col):
        """
        Convierte (fila, col) 2D a un estado único y entero, 1D.
        Ejemplo: Si tienes un mapa de 4x3 (4 filas, 3 columnas)
        Posición (1, 2) → estado = 1 x 3 + 2 = 5
        Posición (0, 0) → estado = 0 x 3 + 0 = 0
        """
        return row * self.ncols + col

    def _state_to_pos(self, state):
        """
        Convierte un estado 1D a (fila, col).
        Ejemplo: Si tienes un mapa de 4x3 (4 filas, 3 columnas)
        Estado 5 en mapa 4x3 → fila = 5 ÷ 3 = 1, columna = 5 % 3 = 2 → (1, 2)
        Estado 0 en mapa 4x3 → fila = 0 ÷ 3 = 0, columna = 0 % 3 = 0 → (0, 0)
        """
        row = state // self.ncols
        col = state % self.ncols
        return row, col

    def reset(self, seed=None, options=None):
        """
        Reinicia el entorno al estado inicial.
        - Resetea la posición del agente a 'S'.
        - Restaura todos los tesoros 'T' consumidos.
        """
        super().reset(seed=seed)

        # 1. Resetear la posición del agente
        self.agent_state = self.start_state
        
        # 2. Resetear el mapa (restaurar tesoros)
        self.current_desc = np.copy(self.base_desc)
        
        # Info estándar de Gymnasium
        info = {}
        
        return self.agent_state, info

    def step(self, action):
        """Ejecuta un paso en el entorno."""
        
        # 1. Obtener posición y tipo de casilla actual
        current_row, current_col = self._state_to_pos(self.agent_state)
        current_tile = self.current_desc[current_row, current_col]
        
        is_slippery = (current_tile == 'W')
        delta = (0, 0)

        # 2. Determinar el delta de movimiento (lógica de deslizamiento)
        if not is_slippery:
            # Movimiento determinista
            delta = self.action_to_delta[action]
        else:
            # Movimiento probabilístico (Agua 'W')
            p = self.np_random.random() # Generador aleatorio de Gym
            
            if action < 4: # Acción Cardinal
                if p < 0.8:
                    delta = self.action_to_delta[action]
                elif p < 0.9:
                    slip_action = self.slip_map_cardinal[action][0] # 90° izq
                    delta = self.action_to_delta[slip_action]
                else:
                    slip_action = self.slip_map_cardinal[action][1] # 90° der
                    delta = self.action_to_delta[slip_action]
            
            else: # Acción Diagonal
                if p < 0.8:
                    delta = self.action_to_delta[action]
                elif p < 0.9:
                    delta = self.slip_map_diagonal[action][0] # Componente 1
                else:
                    delta = self.slip_map_diagonal[action][1] # Componente 2

        # 3. Calcular la nueva posición teórica
        dr, dc = delta
        new_row = current_row + dr
        new_col = current_col + dc

        # 4. Manejar colisiones con los bordes (Clipping)
        # El agente "choca" y se queda en el borde si intenta salir
        new_row = np.clip(new_row, 0, self.nrows - 1)
        new_col = np.clip(new_col, 0, self.ncols - 1)

        # 5. Actualizar el estado del agente
        self.agent_state = self._pos_to_state(new_row, new_col)

        # 6. Calcular recompensa y estado 'done'
        new_tile = self.current_desc[new_row, new_col]
        
        reward = 0
        terminated = False # 'terminated' es para fines del episodio (G, L)
        truncated = False # 'truncated' es para límites de tiempo (no usado aquí)
        
        if new_tile == 'G':
            reward = 10
            terminated = True
        elif new_tile == 'L':
            reward = -10
            terminated = True
        elif new_tile == 'W':
            reward = -1 # Penalización por caer en agua (además de resbalar)
        elif new_tile == 'T':
            reward = 5
            # Consumir el tesoro para este episodio
            self.current_desc[new_row, new_col] = '.'
        
        # Casillas '.', 'S' no dan recompensa (reward = 0)

        info = {}
        
        return self.agent_state, reward, terminated, truncated, info

    def render(self):
        """
        (Futuro) Aquí iría la lógica de Pygame.
        Por ahora, podemos imprimir el mapa.
        """
        # Creamos una copia para mostrar al agente
        render_desc = np.copy(self.current_desc)
        agent_row, agent_col = self._state_to_pos(self.agent_state)
        
        # Marcamos al agente con 'A'
        # (Podríamos usar colores ANSI si quisiéramos)
        render_desc[agent_row, agent_col] = 'A'
        
        print("\n")
        for row in render_desc:
            print(" ".join(row))

    def close(self):
        """(Futuro) Limpieza de recursos de Pygame."""
        pass