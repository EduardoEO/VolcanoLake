import numpy as np

class VolcanoLakeAgent:
    """
    Q-Learning agent for the VolcanoLake environment (modified FrozenLake).
    
    Implements the Q-Learning algorithm with epsilon-greedy policy for exploration.
    Maintains a Q-table that maps states and actions to expected Q-values.
    """
    def __init__(
        self,
        env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """
        Initializes the Q-Learning agent with its hyperparameters.
        
        Args:
            env: Gymnasium environment (must have observation_space and action_space)
            learning_rate (float): Alpha learning rate (controls update speed)
            initial_epsilon (float): Initial exploration value (typically 1.0 = 100%)
            epsilon_decay (float): Amount epsilon decreases after each episode
            final_epsilon (float): Minimum epsilon value (typically 0.1 = 10%)
            discount_factor (float): Gamma discount factor for future rewards (default: 0.95)
        """
        # Creates the Q-table with states and actions, being 16x4
        self.q_values = np.zeros((env.observation_space.n, env.action_space.n)) 
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []

    # With the get_action method, returns a value 0-3 based on the condition
    def get_action(self, env, obs: tuple[int, int, int]) -> int:
        """
        Selects an action using the epsilon-greedy policy.
        
        Exploration vs exploitation strategy:
        - With probability epsilon: random action (EXPLORATION)
        - With probability (1-epsilon): best known action (EXPLOITATION)
        
        Args:
            env: Gymnasium environment (to sample random actions)
            obs (int): Agent's current state (grid position)
            
        Returns:
            int: Selected action (0=left, 1=down, 2=right, 3=up)
        """
        # If the generated number is less than epsilon, generate a random number 0-3 (possible actions). 
        # Therefore, if epsilon is very high, there is a higher probability of exploring and vice versa
        if np.random.random() < self.epsilon: 
            return env.action_space.sample()
        # If the condition is not met, take the value 0-3 that is highest in that observation of our Q-table
        return int(np.argmax(self.q_values[obs])) 

    # With the update method, the Q-table and training error history are updated.
    def update(self, obs, action, reward, terminated, next_obs):
        """
        Updates the Q-table using the Q-Learning Bellman equation.
        
        Update equation:
        Q(s,a) <- Q(s,a) + alpha x [r + gamma x max Q(s',a') - Q(s,a)]
                                   |_______________________|  |______________|
                                          TD Target            Current Value
                                   |_________________________________________|
                                                    TD Error
        
        Args:
            obs (int): Current state
            action (int): Action taken
            reward (float): Reward received
            terminated (bool): If the episode ended
            next_obs (int): Next observed state
        """
        # ===== FUTURE VALUE CALCULATION =====
        # If the episode ended: future_q_value = 0 (no future)
        # If it continues: future_q_value = max_a' Q(s', a') (best possible value in next state)
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        
        # ===== TEMPORAL DIFFERENCE (TD ERROR) CALCULATION =====
        # Measures the difference between prediction and reality
        # TD Error = (r + gamma x max Q(s',a')) - Q(s,a)
        #            |________________________|   |________________|
        #                    TD Target             Current Estimate
        td_error = reward + self.discount_factor * future_q_value - self.q_values[obs][action]

        # TD Error Interpretation:
        # - TD Error > 0: Action was better than expected -> increases Q
        # - TD Error < 0: Action was worse than expected -> decreases Q
        # - TD Error approx 0: Prediction was correct -> minimal change

        # ===== Q-TABLE UPDATE =====
        # Only updates the Q(s,a) cell corresponding to current state-action
        # New Q(s,a) = Old Q(s,a) + alpha x TD Error
        self.q_values[obs][action] += self.lr * td_error

        # ===== ERROR RECORDING FOR ANALYSIS =====
        # Saves the TD error to plot training convergence
        self.training_error.append(td_error)

    def decay_epsilon(self):
        """
        Linearly reduces the epsilon value to decrease exploration.
        
        Called at the end of each episode to implement linear decay
        of the exploration rate. Ensures epsilon never drops below
        the configured minimum (final_epsilon).
        
        Formula: new_epsilon = max(final_epsilon, current_epsilon - decay)
        """
        # Reduces epsilon but never below the established minimum
        # This maintains some exploration even at the end of training
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)