import numpy as np
from gymnasium import Wrapper

class IncreasingLavaHoles(Wrapper):
    """
    Dynamically adds 'L' (Lava) to the map.
    
    The added holes persist between episodes, accumulating
    until reaching 'lim_holes'.
    """
    def __init__(self, env, lim_holes, prob_new_hole=0.05):
        super().__init__(env)
        
        if not hasattr(self.env.unwrapped, "current_desc"):
            raise TypeError("This wrapper only works with VolcanoLakeEnv.")
            
        self.prob_new_hole = prob_new_hole
        self.lim_holes = lim_holes
        
        # Stores all holes created during the entire session.
        self.dynamic_holes = set() 

    def reset(self, **kwargs):
        """
        Resets the base environment AND THEN re-applies all the
        permanent lava holes that have been created so far.
        """
        # 1. Resets the base environment.
        # THIS DELETES THE MAP and restores it from the .csv
        obs, info = self.env.reset(**kwargs)
        
        # 2. Now, we put back all the permanent lava holes
        # that we had saved on the clean 'current_desc' map.
        desc = self.env.unwrapped.current_desc
        for (i, j) in self.dynamic_holes:
            desc[i, j] = 'L'
            
        return obs, info

    def step(self, action):
        """
        Executes the step and, if the limit has not been reached,
        attempts to add a NEW permanent lava hole.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # If the episode has not ended AND we haven't reached the limit yet
        if (not terminated and not truncated and 
            len(self.dynamic_holes) < self.lim_holes and
            self.np_random.random() < self.prob_new_hole):
            
            self._add_random_lava_hole()
        
        return obs, reward, terminated, truncated, info

    def _add_random_lava_hole(self):
        """
        Finds a safe tile ('.' or 'T') and converts it to 'L'.
        This change is permanent (added to self.dynamic_holes).
        """
        desc = self.env.unwrapped.current_desc
        map_shape = desc.shape
        
        valid_positions = []
        for i in range(map_shape[0]):
            for j in range(map_shape[1]):
                # We only convert tiles that are STILL safe
                # (Not 'S', 'G', 'L', 'W' or an existing hole)
                if desc[i, j] in ['.', 'T']:
                    valid_positions.append((i, j))
        
        if valid_positions:
            idx = self.np_random.integers(len(valid_positions))
            i, j = valid_positions[idx]
            
            # 1. We modify the current episode's map
            desc[i, j] = 'L' 
            
            # 2. We save the hole in our permanent 'set'
            self.dynamic_holes.add((i, j))
            
class LimitedVision(Wrapper):
    """
    The agent has a direction (0-7).
    
    - The action taken becomes its new direction.
    - Applies a small reward shaping penalty if the tile in the direction it faces is dangerous ('L' or 'W').
    - Adds the vision to info["vision"].
    """
    
    def __init__(self, env, vision_penalty=-0.1):
        super().__init__(env)
        
        # Check that the environment has action_to_delta
        if not hasattr(self.env.unwrapped, "action_to_delta"):
            raise TypeError("This wrapper only works with an environment that has 'action_to_delta', such as VolcanoLakeEnv.")
            
        # 0-7, matches the env actions
        # We start looking to the right
        self.direction = 1  
        self.penalty = vision_penalty

    def reset(self, **kwargs):
        """ Resets the environment and the agent's direction. """
        obs, info = self.env.reset(**kwargs)
        
        self.direction = 1  # Resets direction, looking to the right
        
        # Adds initial vision to 'info'
        info["vision"] = self._get_vision_in_front(obs)
        
        return obs, info

    def step(self, action):
        """
        Executes the step, applies reward shaping, and updates the direction.
        """
        # The base environment calculates the step result
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # We get the vision (current tile, front tile) in the direction we *were* looking
        vision_tiles = self._get_vision_in_front(obs)
        info["vision"] = vision_tiles
        
        front_tile = vision_tiles[1] # 'L', 'W', 'G', 'T', '.', 'S'

        # Apply reward shaping (Only if the episode has not ended for another reason)
        if not terminated:
            if front_tile == 'L' or front_tile == 'W':
                reward += self.penalty # Adds penalty (-0.1)
                
        # UPDATE the agent's direction. The new direction is the action we just took
        self.direction = action
        
        return obs, reward, terminated, truncated, info

    def _get_vision_in_front(self, obs):
        """
        Returns (current_tile, front_tile) based on the agent's current 'self.direction'.
        """
        desc = self.env.unwrapped.current_desc
        nrows, ncols = desc.shape
        
        # Current agent position
        current_row, current_col = self.env.unwrapped._state_to_pos(obs)
        current_tile = desc[current_row, current_col]

        # Get the delta of the direction we are looking at. We use the base environment's action mapping
        (dr, dc) = self.env.unwrapped.action_to_delta[self.direction]
        
        # Calculate the position of the front tile
        front_row = current_row + dr
        front_col = current_col + dc

        # Check limits
        if not (0 <= front_row < nrows and 0 <= front_col < ncols):
            # Seeing "outside the map" is like seeing Lava
            front_tile = 'L' 
        else:
            front_tile = desc[front_row, front_col]
            
        # Return the letters of the tiles (e.g., '.', 'L')
        return (current_tile, front_tile)

class ActionFlickerWrapper(Wrapper):
    """
    Simulates a "defective controller".
    With a flicker_prob probability, it ignores the action
    chosen by the agent and substitutes it with a random action.
    """
    def __init__(self, env, flicker_prob: float = 0.1):
        super().__init__(env)
        self.flicker_prob = flicker_prob
        
    def step(self, action):
        
        if np.random.rand() < self.flicker_prob:
            flickered_action = self.env.action_space.sample()
            return self.env.step(flickered_action)
        else:
            # The action the agent wanted is executed.
            return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class TorusWrapper(Wrapper):
    """
    "Torus" (Toroidal) World.
    
    When the agent hits an edge or corner (i.e.,
    next_obs == obs_before), it teleports it to the opposite side.
    
    V3 Actions:
    - 0: Up
    - 1: Right
    - 2: Down
    - 3: Left
    - 4: Up-Right
    - 5: Down-Right
    - 6: Down-Left
    - 7: Up-Left
    """
    
    def __init__(self, env):
        super().__init__(env)
        # We get dimensions from the environment
        self.nrows = self.env.unwrapped.nrows
        self.ncols = self.env.unwrapped.ncols

    def step(self, action):
        
        # State BEFORE moving
        obs_before = self.env.unwrapped.agent_state
        
        # Normal V3 environment step
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        
        # If the observation didn't change (we hit a wall) and the episode didn't end
        if (next_obs == obs_before) and not terminated:
            
            # Get current position
            i, j = self.env.unwrapped._state_to_pos(obs_before)
            
            # By default, it doesn't change
            new_state = obs_before 

            # --- Cardinal Logic (Corrected for V3) ---
            # 3=Left at left edge
            if action == 3 and j == 0:  
                new_state = self.env.unwrapped._pos_to_state(i, self.ncols - 1)
            # 1=Right at right edge
            elif action == 1 and j == self.ncols - 1: 
                new_state = self.env.unwrapped._pos_to_state(i, 0)
            # 0=Up at top edge
            elif action == 0 and i == 0: 
                new_state = self.env.unwrapped._pos_to_state(self.nrows - 1, j)
            # 2=Down at bottom edge
            elif action == 2 and i == self.nrows - 1: 
                new_state = self.env.unwrapped._pos_to_state(0, j)

            # --- NEW: Diagonal Logic (Corners only)! ---
            # 7=Up-Left at corner (0,0)
            elif action == 7 and i == 0 and j == 0: 
                new_state = self.env.unwrapped._pos_to_state(self.nrows - 1, self.ncols - 1)
            # 4=Up-Right at corner (0, M-1)
            elif action == 4 and i == 0 and j == self.ncols - 1: 
                new_state = self.env.unwrapped._pos_to_state(self.nrows - 1, 0)
            # 6=Down-Left at corner (N-1, 0)
            elif action == 6 and i == self.nrows - 1 and j == 0: 
                new_state = self.env.unwrapped._pos_to_state(0, self.ncols - 1)
            # 5=Down-Right at corner (N-1, M-1)
            elif action == 5 and i == self.nrows - 1 and j == self.ncols - 1: 
                new_state = self.env.unwrapped._pos_to_state(0, 0)

            # Updates the real internal state of the environment
            self.env.unwrapped.agent_state = new_state
            
            # Overwrites the 'collision' observation with the new one
            next_obs = new_state

        return next_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)