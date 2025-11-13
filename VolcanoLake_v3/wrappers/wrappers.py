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