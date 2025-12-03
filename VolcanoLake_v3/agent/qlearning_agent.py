import numpy as np

class VolcanoLakeAgent:
    def __init__(
        self,
        env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        # Q-table: states x actions (625 x 8 for 25x25 map)
        self.q_values = np.zeros((env.observation_space.n, env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []
    
    def get_action(self, env, obs: int) -> int:
        """
        Selects action using epsilon-greedy policy.
        
        Args:
            env: Environment
            obs: Current state (integer from 0 to 624, default)
        
        Returns:
            Selected action (integer from 0 to 7)
        """
        if np.random.random() < self.epsilon:
            # Exploration: random action (0-7)
            return env.action_space.sample()
        else:
            # Exploitation: best known action
            return int(np.argmax(self.q_values[obs]))

    def update(self, obs, action, reward, terminated, next_obs):
        """
        Updates Q-table using Q-Learning.
        
        Q(s,a) <- Q(s,a) + alpha[r + gamma*max Q(s',a') - Q(s,a)]
        """
        # If the episode ended, there is no future so it will be 0, and if it didn't end it will be 1. 
        # Therefore: 1. If not terminated, future_q_value = max_a' Q(s', a')
        #            2. If terminated, future_q_value = 0
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        
        # Calculate TD error
        td_error = reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        
        # Update Q-value
        self.q_values[obs][action] += self.lr * td_error
        
        # Save error for analysis
        self.training_error.append(td_error)

    def decay_epsilon(self):
        """Reduces epsilon linearly until the minimum."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)