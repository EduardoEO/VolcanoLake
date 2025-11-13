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