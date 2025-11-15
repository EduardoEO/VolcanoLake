import numpy as np
from gymnasium import Wrapper, ObservationWrapper

class IncreasingHoles(Wrapper):
    """
    Wrapper que añade hoyos dinámicos al entorno durante el episodio.
    Los hoyos aparecen con cierta probabilidad en cada paso, haciendo el entorno más desafiante.
    """
    def __init__(self, env, lim_holes, prob_new_hole=0.05):
        """
        Inicializa el wrapper de hoyos crecientes.
        
        Args:
            env: Entorno base de Gymnasium
            lim_holes: Número máximo de hoyos dinámicos permitidos
            prob_new_hole: Probabilidad de crear un nuevo hoyo en cada paso (default: 0.05)
        """
        super().__init__(env)
        self.prob_new_hole = prob_new_hole
        self.lim_holes = lim_holes
        self.dynamic_holes = set() # Conjunto vacío
        
    def step(self, action):
        """
        Ejecuta un paso en el entorno con la lógica de hoyos dinámicos.
        
        Args:
            action: Acción a ejecutar
            
        Returns:
            Tupla (observación, recompensa, terminado, truncado, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        i, j = divmod(obs, self.env.unwrapped.desc.shape[1]) # Calcula la posición (fila, columna) del agente a partir del estado
        if (i, j) in self.dynamic_holes:
            terminated = True # Si el agente cae en un hoyo dinámico, termina el episodio
            reward = 0 # Y la recompensa es 0
    
        # Con cierta probabilidad, añade un hoyo en una casilla aleatoria que no sea meta ni inicio
        if np.random.rand() < self.prob_new_hole:
            self._add_random_hole()
        
        return obs, reward, terminated, truncated, info

    def _add_random_hole(self):
        """
        Añade un hoyo dinámico en una posición aleatoria válida del mapa.
        Solo se añade si no se ha alcanzado el límite máximo de hoyos.
        """
        # Máximo de ciertos hoyos dinámicos, puede ser que le bloquees la salida al agente pero es parte del entrenamiento
        if len(self.dynamic_holes) >= self.lim_holes:
            return
        
        desc = self.env.unwrapped.desc
        valid_positions = [] # Posiciones en F
        for i in range(desc.shape[0]): # Filas
            for j in range(desc.shape[1]): # Columnas
                if desc[i, j] not in [b'S', b'G', b'H']:
                    valid_positions.append((i, j))
        if valid_positions:
            i, j = valid_positions[np.random.randint(len(valid_positions))]
            desc[i, j] = b'H' # Convierte esa casilla en un hoyo
            self.dynamic_holes.add((i, j)) # Añade el punto aleatorio válido al conjunto vacío

class LimitedVisionRewardShaping(Wrapper): # Solo mira la casilla de la derecha, cambiando el self.direcction puedes el agente verá esa casilla y cambiará la recompensa al devolver step
    """
    Wrapper que proporciona visión limitada al agente y modifica las recompensas.
    El agente puede ver solo la casilla actual y una casilla en dirección fija.
    Penaliza suavemente cuando hay un hoyo delante.
    """
    def __init__(self, env):
        """
        Inicializa el wrapper de visión limitada con reward shaping.
        
        Args:
            env: Entorno base de Gymnasium
        """
        super().__init__(env)
        self.direction = 1  # 0=arriba, 1=derecha, 2=abajo, 3=izquierda

    def reset(self, **kwargs):
        """
        Reinicia el entorno y añade información de visión al info.
        
        Returns:
            Tupla (observación_original, info_con_visión)
        """
        obs, info = self.env.reset(**kwargs)
        info["vision"] = self._get_limited_obs(obs)
        return obs, info  # obs sigue siendo un entero

    def step(self, action):
        """
        Ejecuta un paso con reward shaping basado en la visión limitada.
        
        Args:
            action: Acción a ejecutar
            
        Returns:
            Tupla (observación, recompensa_modificada, terminado, truncado, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Obtener visión (pero NO cambiar la observación)
        vision = self._get_limited_obs(obs)
        info["vision"] = vision

        # vision = [actual, delante]
        FRONT = vision[1]

        # Si delante hay un agujero (H = 2), aplicar penalización suave
        if FRONT == 2:
            reward -= 0.1

        return obs, reward, terminated, truncated, info

    def _get_limited_obs(self, obs):
        """
        Calcula la observación de visión limitada del agente.
        
        Args:
            obs: Observación original (posición como entero)
            
        Returns:
            Array numpy con [casilla_actual_codificada, casilla_delante_codificada]
        """
        desc = self.env.unwrapped.desc
        ncols = desc.shape[1]
        i, j = divmod(obs, ncols)
        current = desc[i, j] # pos actual

        # Casilla enfrente según dirección que puede ver
        if self.direction == 0: ni, nj = i-1, j # Casilla arriba
        elif self.direction == 1: ni, nj = i, j+1 # Casilla derecha
        elif self.direction == 2: ni, nj = i+1, j # Casilla abajo
        else: ni, nj = i, j-1 # Casilla izquierda

        # Si está fuera, se considera pared W
        if not (0 <= ni < desc.shape[0] and 0 <= nj < desc.shape[1]):
            front = b'W'
        else:
            front = desc[ni, nj]

        # Codificación S=0, F=1, H=2, G=3, W=4
        def encode(cell):
            if cell == b'S': return 0
            if cell == b'F': return 1
            if cell == b'H': return 2
            if cell == b'G': return 3
            return 4

        return np.array([encode(current), encode(front)])

class LimitedVision(ObservationWrapper): # El agente ve delante dependiendo de la dirección interna (self.direction), que se actualiza según la acción anterior
    """
    Wrapper que proporciona visión limitada dinámica al agente.
    La dirección de visión cambia según las acciones del agente.
    Penaliza cuando hay un hoyo delante y mantiene el espacio de observación original.
    """
    def __init__(self, env):
        """
        Inicializa el wrapper de visión limitada dinámica.
        
        Args:
            env: Entorno base de Gymnasium
        """
        super().__init__(env)
        # 0=arriba, 1=derecha, 2=abajo, 3=izquierda
        self.direction = 1 # Dirección inicial (mirando hacia la derecha por defecto)

        # NO cambiamos observation_space, mantenemos obs original, sino estamos cambiando la Q-table y no podriamos usar el entorno predeterminado 4x4
        self.observation_space = env.observation_space

    def reset(self, **kwargs):
        """
        Reinicia el entorno y la dirección de visión.
        
        Returns:
            Tupla (observación_original, info)
        """
        obs, info = self.env.reset(**kwargs)
        self.direction = 1  # Se resetea siempre mirando a la derecha
        return obs, info

    def step(self, action):
        """
        Ejecuta un paso actualizando la dirección según la acción.
        
        Args:
            action: Acción a ejecutar (0=LEFT, 1=DOWN, 2=RIGHT, 3=UP)
            
        Returns:
            Tupla (observación, recompensa_modificada, terminado, truncado, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Obtiene visión (solo para info, NO sustituye obs)
        vision = self._get_limited_obs(obs)
        info["vision"] = vision  # vision = [actual, delante]

        FRONT = vision[1]  # La casilla delante codificada

        # Si delante hay un hoyo (H == 2), penalizamos suavemente
        if FRONT == 2:
            reward -= 0.01

        # Actualiza dirección según acción
        # 0=LEFT,1=DOWN,2=RIGHT,3=UP
        if action == 3: self.direction = 0  # UP
        elif action == 2: self.direction = 1  # RIGHT
        elif action == 1: self.direction = 2  # DOWN
        elif action == 0: self.direction = 3  # LEFT

        return obs, reward, terminated, truncated, info

    def _get_limited_obs(self, obs):
        """
        Calcula la observación de visión limitada basada en la dirección actual.
        
        Args:
            obs: Observación original (posición como entero)
            
        Returns:
            Array numpy con [casilla_actual_codificada, casilla_delante_codificada]
        """
        desc = self.env.unwrapped.desc
        nrows, ncols = desc.shape
        i, j = divmod(obs, ncols)

        current = desc[i, j] # pos actual

        # Calcula la casilla delante dado self.direction
        if self.direction == 0: ni, nj = i-1, j # arriba
        elif self.direction == 1: ni, nj = i, j+1 # derecha
        elif self.direction == 2: ni, nj = i+1, j # abajo
        else: ni, nj = i, j-1 # izquierda

        # Si está fuera del mapa, cuenta como pared (W)
        if not (0 <= ni < nrows and 0 <= nj < ncols):
            front = b'W'
        else:
            front = desc[ni, nj]

        # Encode: S=0, F=1, H=2, G=3, W=4
        def encode(cell):
            if cell == b'S': return 0
            if cell == b'F': return 1
            if cell == b'H': return 2
            if cell == b'G': return 3
            return 4

        return np.array([encode(current), encode(front)])