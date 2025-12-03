import numpy as np
from gymnasium import Wrapper, ObservationWrapper

class IncreasingHoles(Wrapper):
    """
    Wrapper that adds dynamic holes to the environment during the episode.
    Holes appear with a certain probability at each step, making the environment more challenging.
    """
    def __init__(self, env, lim_holes, prob_new_hole=0.05):
        """
        Initializes the increasing holes wrapper.
        
        Args:
            env: Base Gymnasium environment
            lim_holes: Maximum number of dynamic holes allowed
            prob_new_hole: Probability of creating a new hole at each step (default: 0.05)
        """
        super().__init__(env)
        self.prob_new_hole = prob_new_hole
        self.lim_holes = lim_holes
        # Empty set
        self.dynamic_holes = set() 
        
    def step(self, action):
        """
        Executes a step in the environment with the dynamic holes logic.
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple (observation, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Calculates the agent's position (row, col) from the state
        i, j = divmod(obs, self.env.unwrapped.desc.shape[1]) 
        if (i, j) in self.dynamic_holes:
            # If the agent falls into a dynamic hole, the episode ends
            terminated = True 
            # And the reward is 0
            reward = 0 
    
        # With a certain probability, adds a hole in a random tile that is not goal or start
        if np.random.rand() < self.prob_new_hole:
            self._add_random_hole()
        
        return obs, reward, terminated, truncated, info

    def _add_random_hole(self):
        """
        Adds a dynamic hole in a valid random position on the map.
        It is only added if the maximum limit of holes has not been reached.
        """
        # Maximum of certain dynamic holes, it might block the agent's exit but it's part of training
        if len(self.dynamic_holes) >= self.lim_holes:
            return
        
        desc = self.env.unwrapped.desc
        # Positions in F (Frozen)
        valid_positions = [] 
        # Rows
        for i in range(desc.shape[0]): 
            # Columns
            for j in range(desc.shape[1]): 
                if desc[i, j] not in [b'S', b'G', b'H']:
                    valid_positions.append((i, j))
        if valid_positions:
            i, j = valid_positions[np.random.randint(len(valid_positions))]
            # Converts that tile into a hole
            desc[i, j] = b'H' 
            # Adds the valid random point to the set
            self.dynamic_holes.add((i, j)) 

# Only looks at the right tile, changing self.direction allows the agent to see that tile and will change the reward when returning step
class LimitedVisionRewardShaping(Wrapper): 
    """
    Wrapper that provides limited vision to the agent and modifies rewards.
    The agent can see only the current tile and one tile in a fixed direction.
    Softly penalizes when there is a hole in front.
    """
    def __init__(self, env):
        """
        Initializes the limited vision wrapper with reward shaping.
        
        Args:
            env: Base Gymnasium environment
        """
        super().__init__(env)
        # 0=up, 1=right, 2=down, 3=left
        self.direction = 1  

    def reset(self, **kwargs):
        """
        Resets the environment and adds vision information to info.
        
        Returns:
            Tuple (original_observation, info_with_vision)
        """
        obs, info = self.env.reset(**kwargs)
        info["vision"] = self._get_limited_obs(obs)
        # obs remains an integer
        return obs, info  

    def step(self, action):
        """
        Executes a step with reward shaping based on limited vision.
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple (observation, modified_reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Get vision (but DO NOT change the observation)
        vision = self._get_limited_obs(obs)
        info["vision"] = vision

        # vision = [current, front]
        FRONT = vision[1]

        # If there is a hole in front (H = 2), apply soft penalty
        if FRONT == 2:
            reward -= 0.1

        return obs, reward, terminated, truncated, info

    def _get_limited_obs(self, obs):
        """
        Calculates the agent's limited vision observation.
        
        Args:
            obs: Original observation (position as integer)
            
        Returns:
            Numpy array with [encoded_current_tile, encoded_front_tile]
        """
        desc = self.env.unwrapped.desc
        ncols = desc.shape[1]
        i, j = divmod(obs, ncols)
        # current pos
        current = desc[i, j] 

        # Tile in front according to the direction it can see
        # Tile above
        if self.direction == 0: ni, nj = i-1, j 
        # Tile right
        elif self.direction == 1: ni, nj = i, j+1 
        # Tile below
        elif self.direction == 2: ni, nj = i+1, j 
        # Tile left
        else: ni, nj = i, j-1 

        # If outside, considered wall W
        if not (0 <= ni < desc.shape[0] and 0 <= nj < desc.shape[1]):
            front = b'W'
        else:
            front = desc[ni, nj]

        # Encoding S=0, F=1, H=2, G=3, W=4
        def encode(cell):
            if cell == b'S': return 0
            if cell == b'F': return 1
            if cell == b'H': return 2
            if cell == b'G': return 3
            return 4

        return np.array([encode(current), encode(front)])

# The agent sees in front depending on internal direction (self.direction), which updates according to previous action
class LimitedVision(ObservationWrapper): 
    """
    Wrapper that provides dynamic limited vision to the agent.
    The vision direction changes according to the agent's actions.
    Penalizes when there is a hole in front and keeps the original observation space.
    """
    def __init__(self, env):
        """
        Initializes the dynamic limited vision wrapper.
        
        Args:
            env: Base Gymnasium environment
        """
        super().__init__(env)
        # 0=up, 1=right, 2=down, 3=left
        # Initial direction (looking right by default)
        self.direction = 1 

        # WE DO NOT change observation_space, we keep original obs, otherwise we are changing the Q-table and couldn't use the default 4x4 env
        self.observation_space = env.observation_space

    def reset(self, **kwargs):
        """
        Resets the environment and the vision direction.
        
        Returns:
            Tuple (original_observation, info)
        """
        obs, info = self.env.reset(**kwargs)
        # Always resets looking to the right
        self.direction = 1  
        return obs, info

    def step(self, action):
        """
        Executes a step updating the direction according to the action.
        
        Args:
            action: Action to execute (0=LEFT, 1=DOWN, 2=RIGHT, 3=UP)
            
        Returns:
            Tuple (observation, modified_reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Gets vision (only for info, DOES NOT substitute obs)
        vision = self._get_limited_obs(obs)
        # vision = [current, front]
        info["vision"] = vision  

        # The encoded front tile
        FRONT = vision[1]  

        # If there is a hole in front (H == 2), we penalize softly
        if FRONT == 2:
            reward -= 0.01

        # Updates direction according to action
        # 0=LEFT,1=DOWN,2=RIGHT,3=UP
        # UP
        if action == 3: self.direction = 0  
        # RIGHT
        elif action == 2: self.direction = 1  
        # DOWN
        elif action == 1: self.direction = 2  
        # LEFT
        elif action == 0: self.direction = 3  

        return obs, reward, terminated, truncated, info

    def _get_limited_obs(self, obs):
        """
        Calculates the limited vision observation based on the current direction.
        
        Args:
            obs: Original observation (position as integer)
            
        Returns:
            Numpy array with [encoded_current_tile, encoded_front_tile]
        """
        desc = self.env.unwrapped.desc
        nrows, ncols = desc.shape
        i, j = divmod(obs, ncols)

        # current pos
        current = desc[i, j] 

        # Calculates the front tile given self.direction
        # up
        if self.direction == 0: ni, nj = i-1, j 
        # right
        elif self.direction == 1: ni, nj = i, j+1 
        # below
        elif self.direction == 2: ni, nj = i+1, j 
        # left
        else: ni, nj = i, j-1 

        # If outside the map, counts as wall (W)
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