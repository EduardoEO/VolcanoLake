import numpy as np
import os

def generate_map(size, tile_proportions, start_pos=(0, 0), goal_pos=None):
    """
    Generates a random map for VolcanoLake.
    
    Args:
        size (int): The size of the map (size x size).
        tile_proportions (dict): Proportions of 'L', 'W', 'T', and '.'.
        start_pos (tuple): Position (row, col) for 'S'.
        goal_pos (tuple): Position (row, col) for 'G'.
    """
    
    # If the goal is not specified, place it in the opposite corner
    if goal_pos is None:
        goal_pos = (size - 1, size - 1)

    # 1. Calculate the total number of tiles
    total_tiles = size * size
    
    # 2. Create the list of tiles (including S and G)
    tiles = []
    
    num_lava = int(total_tiles * tile_proportions.get('L', 0))
    num_water = int(total_tiles * tile_proportions.get('W', 0))
    num_treasure = int(total_tiles * tile_proportions.get('T', 0))
    
    # The rest is ground
    num_ground = total_tiles - num_lava - num_water - num_treasure
    
    tiles.extend(['L'] * num_lava)
    tiles.extend(['W'] * num_water)
    tiles.extend(['T'] * num_treasure)
    tiles.extend(['.'] * num_ground)
    
    # 3. Randomly shuffle the tiles
    np.random.shuffle(tiles)
    
    # 4. Create the 2D map
    map_grid = np.array(tiles).reshape((size, size))
    
    # 5. Force S and G into their positions
    # Note: The tile that was here (L, W, T, .) will be replaced
    map_grid[start_pos[0], start_pos[1]] = 'S'
    map_grid[goal_pos[0], goal_pos[1]] = 'G'
            
    return map_grid

def save_map_to_csv(map_array, output_path):
    """Saves the map array to a CSV file."""
    np.savetxt(output_path, map_array, delimiter=',', fmt='%s')

# --- Main Block ---
if __name__ == "__main__":
    
    # --- Configuration ---
    # Proportions of special tiles
    # (Adjust this to your liking)
    PROPORTIONS = {
        # 10% Lava
        'L': 0.10,
        # 15% Water
        'W': 0.15,
        # 5% Treasure
        'T': 0.05,
        # The remaining 70% will be '.' (Ground)
    }
    
    # Sizes of the maps to generate
    MAP_SIZES = [5, 25, 50, 100]
    
    # Output folder path (relative to the v3 project root)
    # Assumes this script is in v3/scripts/ and maps go in v3/maps/
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'maps')
    
    # --- Execution ---
    
    print(f"Creando carpeta de mapas en: {os.path.abspath(OUTPUT_DIR)}")
    # Create the output folder if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for size in MAP_SIZES:
        print(f"Generando mapa {size}x{size}...")
        
        # 1. Generate the map
        game_map = generate_map(size, PROPORTIONS)
        
        # 2. Define output file path
        file_name = f"map_{size}x{size}.csv"
        output_path = os.path.join(OUTPUT_DIR, file_name)
        
        # 3. Save the map
        save_map_to_csv(game_map, output_path)
        print(f"Mapa guardado en: {output_path}")

    print("\n¡Generación de mapas completada!")