import numpy as np
import os

def generate_map(size, tile_proportions, start_pos=(0, 0), goal_pos=None):
    """
    Genera un mapa aleatorio para VolcanoLake.
    
    Args:
        size (int): El tamaño del mapa (size x size).
        tile_proportions (dict): Proporciones de 'L', 'W', 'T', y '.'.
        start_pos (tuple): Posición (fila, col) para 'S'.
        goal_pos (tuple): Posición (fila, col) para 'G'.
    """
    
    # Si no se especifica la meta, se pone en la esquina opuesta
    if goal_pos is None:
        goal_pos = (size - 1, size - 1)

    # 1. Calcular el número total de casillas
    total_tiles = size * size
    
    # 2. Crear la lista de casillas (incluyendo S y G)
    tiles = []
    
    num_lava = int(total_tiles * tile_proportions.get('L', 0))
    num_water = int(total_tiles * tile_proportions.get('W', 0))
    num_treasure = int(total_tiles * tile_proportions.get('T', 0))
    
    # El resto es tierra
    num_ground = total_tiles - num_lava - num_water - num_treasure
    
    tiles.extend(['L'] * num_lava)
    tiles.extend(['W'] * num_water)
    tiles.extend(['T'] * num_treasure)
    tiles.extend(['.'] * num_ground)
    
    # 3. Mezclar aleatoriamente las casillas
    np.random.shuffle(tiles)
    
    # 4. Crear el mapa 2D
    map_grid = np.array(tiles).reshape((size, size))
    
    # 5. Forzar S y G en sus posiciones
    # Nota: La casilla que estuviera aquí (L, W, T, .) será reemplazada
    map_grid[start_pos[0], start_pos[1]] = 'S'
    map_grid[goal_pos[0], goal_pos[1]] = 'G'
            
    return map_grid

def save_map_to_csv(map_array, output_path):
    """Guarda el array del mapa en un archivo CSV."""
    np.savetxt(output_path, map_array, delimiter=',', fmt='%s')

# --- Bloque Principal ---
if __name__ == "__main__":
    
    # --- Configuración ---
    
    # Proporciones de las casillas especiales
    # (Ajusta esto a tu gusto)
    PROPORTIONS = {
        'L': 0.10,  # 10% Lava
        'W': 0.15,  # 15% Agua
        'T': 0.05,  # 5% Tesoro
        # El 70% restante será '.' (Tierra)
    }
    
    # Tamaños de los mapas a generar
    MAP_SIZES = [5, 25, 50, 100]
    
    # Ruta de la carpeta de salida (relativa a la raíz del proyecto v3)
    # Asume que este script está en v3/scripts/ y los mapas van en v3/maps/
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'maps')
    
    # --- Ejecución ---
    
    print(f"Creando carpeta de mapas en: {os.path.abspath(OUTPUT_DIR)}")
    # Crear la carpeta de salida si no existe
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for size in MAP_SIZES:
        print(f"Generando mapa {size}x{size}...")
        
        # 1. Generar el mapa
        game_map = generate_map(size, PROPORTIONS)
        
        # 2. Definir ruta del archivo de salida
        file_name = f"map_{size}x{size}.csv"
        output_path = os.path.join(OUTPUT_DIR, file_name)
        
        # 3. Guardar el mapa
        save_map_to_csv(game_map, output_path)
        print(f"Mapa guardado en: {output_path}")

    print("\n¡Generación de mapas completada!")