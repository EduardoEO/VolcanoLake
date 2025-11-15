import numpy as np
from gymnasium import Wrapper

class IncreasingLavaHoles(Wrapper):
    """
    Añade 'L' (Lava) dinámicamente al mapa.
    
    Los hoyos añadidos persisten entre episodios, acumulándose
    hasta llegar al 'lim_holes'.
    """
    def __init__(self, env, lim_holes, prob_new_hole=0.05):
        super().__init__(env)
        
        if not hasattr(self.env.unwrapped, "current_desc"):
            raise TypeError("Este wrapper solo funciona con VolcanoLakeEnv.")
            
        self.prob_new_hole = prob_new_hole
        self.lim_holes = lim_holes
        
        # Almacena todos los hoyos creados durante toda la sesión.
        self.dynamic_holes = set() 

    def reset(self, **kwargs):
        """
        Resetea el entorno base Y LUEGO re-aplica todos los
        hoyos de lava permanentes que se han creado hasta ahora.
        """
        # 1. Resetea el entorno base.
        # ESTO BORRA EL MAPA y lo restaura desde el .csv
        obs, info = self.env.reset(**kwargs)
        
        # 2. Ahora, volvemos a poner todos los hoyos de lava permanentes
        # que teníamos guardados en el mapa 'current_desc' limpio.
        desc = self.env.unwrapped.current_desc
        for (i, j) in self.dynamic_holes:
            desc[i, j] = 'L'
            
        return obs, info

    def step(self, action):
        """
        Ejecuta el paso y, si no se ha alcanzado el límite,
        intenta añadir un NUEVO hoyo de lava permanente.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Si el episodio no ha terminado Y aún no hemos llegado al límite
        if (not terminated and not truncated and 
            len(self.dynamic_holes) < self.lim_holes and
            self.np_random.random() < self.prob_new_hole):
            
            self._add_random_lava_hole()
        
        return obs, reward, terminated, truncated, info

    def _add_random_lava_hole(self):
        """
        Encuentra una casilla segura ('.' o 'T') y la convierte en 'L'.
        Este cambio es permanente (se añade a self.dynamic_holes).
        """
        desc = self.env.unwrapped.current_desc
        map_shape = desc.shape
        
        valid_positions = []
        for i in range(map_shape[0]):
            for j in range(map_shape[1]):
                # Solo convertimos casillas que AÚN son seguras
                # (No 'S', 'G', 'L', 'W' o un hoyo ya existente)
                if desc[i, j] in ['.', 'T']:
                    valid_positions.append((i, j))
        
        if valid_positions:
            idx = self.np_random.integers(len(valid_positions))
            i, j = valid_positions[idx]
            
            # 1. Modificamos el mapa del episodio actual
            desc[i, j] = 'L' 
            
            # 2. Guardamos el hoyo en nuestro 'set' permanente
            self.dynamic_holes.add((i, j))
            
class LimitedVision(Wrapper):
    """
    El agente tiene una dirección (0-7).
    
    - La acción que toma se convierte en su nueva dirección.
    - Aplica una pequeña penalización de reward shaping si la casilla en la dirección que mira es peligrosa ('L' o 'W').
    - Añade la visión a info["vision"].
    """
    
    def __init__(self, env, vision_penalty=-0.1):
        super().__init__(env)
        
        # Comprobamos que el entorno tiene action_to_delta
        if not hasattr(self.env.unwrapped, "action_to_delta"):
            raise TypeError("Este wrapper solo funciona con un entorno que tenga 'action_to_delta', como VolcanoLakeEnv.")
            
        # 0-7, coincide con las acciones del env
        self.direction = 1  # Empezamos mirando a la derecha
        self.penalty = vision_penalty

    def reset(self, **kwargs):
        """ Resetea el entorno y la dirección del agente. """
        obs, info = self.env.reset(**kwargs)
        
        # Resetea la dirección
        self.direction = 1  # Mirando a la derecha
        
        # Añade la visión inicial al 'info'
        info["vision"] = self._get_vision_in_front(obs)
        
        return obs, info

    def step(self, action):
        """
        Ejecuta el paso, aplica el reward shaping y actualiza la dirección.
        """
        # El entorno base calcula el resultado del paso
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Obtenemos la visión (casilla actual, casilla enfrente) en la dirección que *teníamos*
        vision_tiles = self._get_vision_in_front(obs)
        info["vision"] = vision_tiles
        
        front_tile = vision_tiles[1] # 'L', 'W', 'G', 'T', '.', 'S'

        # Aplicamos el reward shaping (Solo si el episodio no ha terminado por otra razón)
        if not terminated:
            if front_tile == 'L' or front_tile == 'W':
                reward += self.penalty # Añade penalización (-0.1)
                
        # ACTUALIZAMOS la dirección del agente. La nueva dirección es la acción que acabamos de tomar
        self.direction = action
        
        return obs, reward, terminated, truncated, info

    def _get_vision_in_front(self, obs):
        """
        Devuelve (casilla_actual, casilla_enfrente) basándose
        en la 'self.direction' actual del agente.
        """
        desc = self.env.unwrapped.current_desc
        nrows, ncols = desc.shape
        
        # Posición actual del agente
        current_row, current_col = self.env.unwrapped._state_to_pos(obs)
        current_tile = desc[current_row, current_col]

        # Obtener el delta de la dirección en la que miramos. Usamos el mapeo de acciones del entorno base
        (dr, dc) = self.env.unwrapped.action_to_delta[self.direction]
        
        # Calcular la posición de la casilla de enfrente
        front_row = current_row + dr
        front_col = current_col + dc

        # Comprobar límites
        if not (0 <= front_row < nrows and 0 <= front_col < ncols):
            front_tile = 'L' # Ver "fuera del mapa" es como ver Lava
        else:
            front_tile = desc[front_row, front_col]
            
        # Devolvemos las letras de las casillas (ej. '.', 'L')
        return (current_tile, front_tile)

class ActionFlickerWrapper(Wrapper):
    """
    Simula un "controlador defectuoso".
    Con una probabilidad flicker_prob, ignora la acción
    elegida por el agente y la sustituye por una acción aleatoria.
    """
    def __init__(self, env, flicker_prob: float = 0.1):
        super().__init__(env)
        self.flicker_prob = flicker_prob
        
    def step(self, action):
        
        if np.random.rand() < self.flicker_prob:
            flickered_action = self.env.action_space.sample()
            return self.env.step(flickered_action)
        else:
            # Se ejecuta la acción que el agente quería.
            return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class TorusWrapper(Wrapper):
    """
    Mundo "Toro" (Toroidal).
    
    Cuando el agente choca contra un borde o esquina (es decir,
    next_obs == obs_before), lo teletransporta al lado opuesto.
    
    Acciones V3:
    - 0: Arriba
    - 1: Derecha
    - 2: Abajo
    - 3: Izquierda
    - 4: Arriba-Derecha
    - 5: Abajo-Derecha
    - 6: Abajo-Izquierda
    - 7: Arriba-Izquierda
    """
    
    def __init__(self, env):
        super().__init__(env)
        # Obtenemos las dimensiones desde el entorno V3
        self.nrows = self.env.unwrapped.nrows
        self.ncols = self.env.unwrapped.ncols

    def step(self, action):
        
        # Estado ANTES de moverse
        obs_before = self.env.unwrapped.agent_state
        
        # Paso normal del entorno V3
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Si la observación no cambió (chocamos) y el episodio no terminó
        if (next_obs == obs_before) and not terminated:
            
            # Obtenemos la posición actual
            i, j = self.env.unwrapped._state_to_pos(obs_before)
            
            new_state = obs_before # Por defecto, no cambia

            # --- Lógica Cardinal (Corregida para V3) ---
            if action == 3 and j == 0:  # 3=Izquierda en borde izquierdo
                new_state = self.env.unwrapped._pos_to_state(i, self.ncols - 1)
            elif action == 1 and j == self.ncols - 1: # 1=Derecha en borde derecho
                new_state = self.env.unwrapped._pos_to_state(i, 0)
            elif action == 0 and i == 0: # 0=Arriba en borde superior
                new_state = self.env.unwrapped._pos_to_state(self.nrows - 1, j)
            elif action == 2 and i == self.nrows - 1: # 2=Abajo en borde inferior
                new_state = self.env.unwrapped._pos_to_state(0, j)

            # --- ¡NUEVO: Lógica Diagonal (Solo esquinas)! ---
            elif action == 7 and i == 0 and j == 0: # 7=Arriba-Izq en esquina (0,0)
                new_state = self.env.unwrapped._pos_to_state(self.nrows - 1, self.ncols - 1)
            elif action == 4 and i == 0 and j == self.ncols - 1: # 4=Arriba-Der en esquina (0, M-1)
                new_state = self.env.unwrapped._pos_to_state(self.nrows - 1, 0)
            elif action == 6 and i == self.nrows - 1 and j == 0: # 6=Abajo-Izq en esquina (N-1, 0)
                new_state = self.env.unwrapped._pos_to_state(0, self.ncols - 1)
            elif action == 5 and i == self.nrows - 1 and j == self.ncols - 1: # 5=Abajo-Der en esquina (N-1, M-1)
                new_state = self.env.unwrapped._pos_to_state(0, 0)

            # Actualiza el estado interno real del entorno
            # ¡CORREGIDO para V3!
            self.env.unwrapped.agent_state = new_state
            
            # Sobrescribe la observación de "choque" con la nueva
            next_obs = new_state

        return next_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)