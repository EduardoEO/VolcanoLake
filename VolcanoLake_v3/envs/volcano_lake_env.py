import gymnasium as gym
import numpy as np
import os
import pygame

class VolcanoLakeEnv(gym.Env):
    """
    Custom VolcanoLake_v3 environment.
    
    Initial clarifications:
    Rules:
    - S: Start
    - G: Goal (+10, terminates)
    - L: Lava (-10, terminates)
    - W: Water (-1, slips)
    - T: Treasure (+5, continues, is consumed)
    - .: Land (0)
    
    Actions (8):
    - 0: Up
    - 1: Right
    - 2: Down
    - 3: Left
    - 4: Up-Right
    - 5: Down-Right
    - 6: Down-Left
    - 7: Up-Left
    
    Water Logic (W):
    - 80% success.
    - 10% slip 90° left (cardinal) or component 1 (diagonal).
    - 10% slip 90° right (cardinal) or component 2 (diagonal).
    """
    
    # Optional: metadata for render mode
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, map_file_path=None, render_mode=None):
        """
        Initializes the VolcanoLake v3 environment.
        
        Args:
            map_file_path (str, optional): Path to the map CSV file. 
                                           If None, uses default map (map_25x25.csv).
            render_mode (str, optional): Rendering mode ('human', 'rgb_array', or None).
                                         - 'human': Shows graphic window on screen
                                         - 'rgb_array': Returns arrays for video recording
                                         - None: Does not render
        """
        super().__init__()

        # Default map
        if map_file_path is None:
            current_dir = os.path.dirname(__file__)
            map_file_path = os.path.join(current_dir, '..', 'map_25x25.csv')            
        
        # --- Map Loading ---
        if not os.path.exists(map_file_path):
            raise FileNotFoundError(f"No se pudo encontrar el archivo del mapa: {map_file_path}")
            
        # Load base map (immutable)
        # Use 'U' for strings (Unicode)
        self.base_desc = np.genfromtxt(map_file_path, dtype='<U1', delimiter=',')
        
        # Create a mutable copy for the current session (to consume treasures)
        self.current_desc = np.copy(self.base_desc)
        
        self.nrows, self.ncols = self.base_desc.shape

        # --- Space Definition ---
        # The observation space is a single integer representing the position
        self.observation_space = gym.spaces.Discrete(self.nrows * self.ncols)
        
        # The action space has 8 actions
        self.action_space = gym.spaces.Discrete(8)

        # --- Action Mapping ---
        # (delta_row, delta_col)
        self.action_to_delta = {
            # Up
            0: (-1, 0),
            # Right
            1: (0, 1),
            # Down
            2: (1, 0),
            # Left
            3: (0, -1),
            # Up-Right
            4: (-1, 1),
            # Down-Right
            5: (1, 1),
            # Down-Left
            6: (1, -1),
            # Up-Left
            7: (-1, -1)
        }
        
        # --- Slippage Logic (Water 'W') ---
        # For cardinal actions (0-3), maps to 90° left/right actions
        self.slip_map_cardinal = {
            # Up -> Left, Right
            0: (3, 1),
            # Right -> Up, Down
            1: (0, 2),
            # Down -> Right, Left
            2: (1, 3),
            # Left -> Down, Up
            3: (2, 0)
        }
        
        # For diagonal actions (4-7), maps to the *deltas* of their components
        self.slip_map_diagonal = {
            # Up-Right -> Up, Right
            4: ((-1, 0), (0, 1)),
            # Down-Right -> Down, Right
            5: ((1, 0), (0, 1)),
            # Down-Left -> Down, Left
            6: ((1, 0), (0, -1)),
            # Up-Left -> Up, Left
            7: ((-1, 0), (0, -1))
        }

        # --- Initial State ---
        # Find the 'S'
        start_pos_array = np.where(self.base_desc == 'S')
        if start_pos_array[0].size == 0:
            raise ValueError("El mapa no contiene un punto de inicio 'S'")
        
        self.start_pos = (start_pos_array[0][0], start_pos_array[1][0])
        self.start_state = self._pos_to_state(self.start_pos[0], self.start_pos[1])
        
        # The agent's current state (will be reset in reset())
        self.agent_state = self.start_state
        
        # --- Statistics Tracking ---
        # Treasure history per episode
        self.treasures_found_queue = []
        # Current episode counter
        self.current_episode_treasures = 0
        
        # Success history per episode
        self.success_queue = []
        # Flag for current episode
        self.current_episode_success = False
        
        self.just_initialized = True
    
        # --- Render Configuration ---
        # Store the mode
        self.render_mode = render_mode
        # Pygame window (created in render())
        self.window = None
        # Pygame clock
        self.clock = None
        
        # Cell size in pixels
        if self.nrows > 50 or self.ncols > 50:
            self.cell_size = 16 
        elif self.nrows > 20 or self.ncols > 20:
            self.cell_size = 32
        else:
            self.cell_size = 64
            
        self.window_width = self.ncols * self.cell_size
        self.window_height = self.nrows * self.cell_size
        
        # RGB color definition
        self.colors = {
            # Green (Start)
            'S': (60, 179, 113),
            # Golden (Goal)
            'G': (255, 215, 0),
            # Red (Lava)
            'L': (220, 20, 60),
            # Blue (Water)
            'W': (30, 144, 255),
            # Violet (Treasure)
            'T': (148, 0, 211),
            # Brown (Land)
            '.': (205, 133, 63),
            # Light Grey (Agent)
            'AGENT': (220, 220, 220)
        }


    def _pos_to_state(self, row, col):
        """
        Converts 2D (row, col) to a unique 1D state integer.
        Example: If you have a 4x3 map (4 rows, 3 columns)
        Position (1, 2) -> state = 1 x 3 + 2 = 5
        Position (0, 0) -> state = 0 x 3 + 0 = 0
        """
        return row * self.ncols + col

    def _state_to_pos(self, state):
        """
        Converts a 1D state to (row, col).
        Example: If you have a 4x3 map (4 rows, 3 columns)
        State 5 in 4x3 map -> row = 5 // 3 = 1, col = 5 % 3 = 2 -> (1, 2)
        State 0 in 4x3 map -> row = 0 // 3 = 0, col = 0 % 3 = 0 -> (0, 0)
        """
        row = state // self.ncols
        col = state % self.ncols
        return row, col

    def reset(self, seed=None, options=None):
        """
        Resets the environment to the initial state to start a new episode.
        
        This function performs the following tasks:
        1. Saves statistics from the previous episode (treasures and success)
        2. Resets counters for the new episode
        3. Restores the agent's position to the start point 'S'
        4. Restores all treasures 'T' consumed during the previous episode
        
        Args:
            seed (int, optional): Seed for the random number generator.
                                  Useful for reproducibility in experiments.
            options (dict, optional): Additional options (reserved for extensions).
        
        Returns:
            tuple: (observation, info)
                - observation (int): Initial agent state (position of 'S')
                - info (dict): Empty dictionary with additional info
        
        Note:
            The first time it is called (from __init__), it DOES NOT save statistics
            because no episode has finished yet.
        """
        super().reset(seed=seed)
        
        # Only save data if it is NOT the first call (the init one)
        if not self.just_initialized:
            # 1. Save treasures from previous episode
            self.treasures_found_queue.append(self.current_episode_treasures)
            
            # 2. Save success of previous episode
            success_value = 1.0 if self.current_episode_success else 0.0
            self.success_queue.append(success_value)
        
        self.just_initialized = False
        
        # 2. Reset treasure counter for new episode
        self.current_episode_treasures = 0
        # Reset the flag too
        self.current_episode_success = False

        # 3. Reset agent position
        self.agent_state = self.start_state
        
        # 4. Reset map (restore treasures)
        self.current_desc = np.copy(self.base_desc)
        
        # Standard Gymnasium info
        info = {}
        
        return self.agent_state, info

    def step(self, action):
        """
        Executes an action in the environment and returns the result.
        
        This is the core of the environment. It processes the agent's action following these steps:
        1. Determines if there is slippage based on the current tile type
        2. Calculates the resulting movement (may be different if it slips)
        3. Updates the agent's position (with clipping at edges)
        4. Calculates rewards based on the new tile
        5. Determines if the episode ended
        
        Args:
            action (int): Action to execute (0-7)
                          0: Up, 1: Right, 2: Down, 3: Left
                          4: Up-Right, 5: Down-Right
                          6: Down-Left, 7: Up-Left
        
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
                - observation (int): New agent state
                - reward (float): Reward obtained in this step
                - terminated (bool): True if the episode ended (G or L reached)
                - truncated (bool): True if time limit reached (not used here)
                - info (dict): Additional information (empty for now)
        
        Slippage mechanics in Water ('W'):
        - Cardinal actions (0-3): 80% success, 10% slip 90° left, 10% slip 90° right
        - Diagonal actions (4-7): 80% success, 10% vertical component, 10% horizontal component
        
        Reward system:
        - Goal 'G': +10 (successful episode ends)
        - Lava 'L': -10 (failed episode ends)
        - Treasure 'T': +5 (consumed and disappears)
        - Water 'W': -1 (penalty for getting wet)
        - Land '.' / Start 'S': -0.001 (small cost to encourage efficiency)
        """
        info = {}
        
        # 1. Get current position and tile type
        current_row, current_col = self._state_to_pos(self.agent_state)
        current_tile = self.current_desc[current_row, current_col]
        
        is_slippery = (current_tile == 'W')
        delta = (0, 0)

        # 2. Determine movement delta (slippage logic)
        if not is_slippery:
            # Deterministic movement
            delta = self.action_to_delta[action]
        else:
            # Probabilistic movement (Water 'W')
            # Gym random generator
            p = self.np_random.random()
            
            # Cardinal Action
            if action < 4:
                if p < 0.8:
                    delta = self.action_to_delta[action]
                elif p < 0.9:
                    # 90° left
                    slip_action = self.slip_map_cardinal[action][0]
                    delta = self.action_to_delta[slip_action]
                else:
                    # 90° right
                    slip_action = self.slip_map_cardinal[action][1]
                    delta = self.action_to_delta[slip_action]
            
            # Diagonal Action
            else:
                if p < 0.8:
                    delta = self.action_to_delta[action]
                elif p < 0.9:
                    # Component 1
                    delta = self.slip_map_diagonal[action][0]
                else:
                    # Component 2
                    delta = self.slip_map_diagonal[action][1]

        # 3. Calculate theoretical new position
        dr, dc = delta
        new_row = current_row + dr
        new_col = current_col + dc

        # 4. Handle edge collisions (Clipping)
        # The agent "hits" and stays at the edge if it tries to exit
        new_row = np.clip(new_row, 0, self.nrows - 1)
        new_col = np.clip(new_col, 0, self.ncols - 1)

        # 5. Update agent state
        self.agent_state = self._pos_to_state(new_row, new_col)

        # 6. Calculate reward and 'done' state
        new_tile = self.current_desc[new_row, new_col]
        
        # Prevent the agent from learning to only go for T instead of reaching G
        reward = -0.001
        # 'terminated' is for episode ends (G, L)
        terminated = False
        # 'truncated' is for time limits (not used here)
        truncated = False
        
        if new_tile == 'G':
            reward = 10
            terminated = True
            self.current_episode_success = True
        elif new_tile == 'L':
            reward = -10
            terminated = True
        elif new_tile == 'W':
            # Penalty for falling into water (besides slipping)
            reward = -1
        elif new_tile == 'T':
            reward = 5
            # Consume treasure for this episode
            self.current_desc[new_row, new_col] = '.'
            # Increment counter
            self.current_episode_treasures += 1
        # Tiles '.', 'S' give no reward (reward = 0)
        
        return self.agent_state, reward, terminated, truncated, info

    def render(self):
        """
        Renders the VolcanoLake environment using Pygame.
    
        Behavior according to render mode:
        - "human": Shows the graphic window on screen for interactive visualization
        - "rgb_array": Returns a NumPy array with pixels for video recording
        - None: Does not render anything (silent mode)
    
        The rendering includes:
        - All map tiles with their corresponding colors
        - The agent represented as a light grey rectangle
    
        Returns:
        numpy.ndarray or None: Pixel array (height, width, 3) if mode="rgb_array", 
                               None otherwise
        """
        if self.render_mode is None:
            return
        
        if self.window is None:
            # Only initialized if render() is called for the first time
            pygame.init()
            pygame.display.set_caption("VolcanoLake_v3")
            
            if self.render_mode == "human":
                # Create a visible window
                self.window = pygame.display.set_mode((self.window_width, self.window_height))
            elif self.render_mode == "rgb_array":
                # Create a hidden surface 
                self.window = pygame.Surface((self.window_width, self.window_height))
            
        if self.clock is None:
            self.clock = pygame.time.Clock()
            
        # --- Drawing Logic ---
        
        # Create a canvas for drawing
        canvas = pygame.Surface((self.window_width, self.window_height))
        # White background just in case
        canvas.fill((255, 255, 255))

        # Draw all map tiles
        for r in range(self.nrows):
            for c in range(self.ncols):
                tile_type = self.current_desc[r, c]
                # Black if error
                color = self.colors.get(tile_type, (0, 0, 0))
                
                pygame.draw.rect(
                    canvas,
                    color,
                    (c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                )

        # Draw the agent on top
        agent_row, agent_col = self._state_to_pos(self.agent_state)
        pygame.draw.rect(
            canvas,
            self.colors['AGENT'],
            (agent_col * self.cell_size, agent_row * self.cell_size, self.cell_size, self.cell_size)
        )

        # --- Output Logic ---
        
        if self.render_mode == "human":
            # If "human" mode, copy canvas to window
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            
            # Control FPS
            self.clock.tick(self.metadata["render_fps"])
            
        elif self.render_mode == "rgb_array":
            # If "rgb_array", copy canvas to hidden surface
            self.window.blit(canvas, (0, 0))
            
            # Return content as a NumPy array
            # Transpose from (width, height, 3) to (height, width, 3)
            return np.transpose(
                pygame.surfarray.array3d(self.window), (1, 0, 2)
            )
            

    def close(self):
        """
        Cleans and closes all Pygame resources.
    
        It is important to call this function at the end of training or usage
        to free memory and avoid issues with multiple initializations.
    
        Actions performed:
        - Closes the Pygame window (if it exists)
        - Closes Pygame completely
        - Resets window and clock variables to None
        
        Note: If not called, windows may remain open in the background
        consuming memory and causing warnings on subsequent restarts.
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            # Important for reset
            self.window = None
            self.clock = None