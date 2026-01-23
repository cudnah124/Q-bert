import numpy as np
from collections import deque


class MultiStepBuffer:
    """
    Buffer for computing n-step returns.
    
    Stores the last n transitions and computes:
    R_t^(n) = r_t + γ*r_{t+1} + γ^2*r_{t+2} + ... + γ^{n-1}*r_{t+n-1}
    
    Used in Rainbow DQN for multi-step learning.
    
    Paper: Mnih et al. 2016 - "Asynchronous Methods for Deep Reinforcement Learning"
    https://arxiv.org/abs/1602.01783
    """
    
    def __init__(self, n_steps=3, gamma=0.99):
        """
        Args:
            n_steps: Number of steps for n-step return
            gamma: Discount factor
        """
        self.n_steps = n_steps
        self.gamma = gamma
        self.buffer = deque(maxlen=n_steps)
    
    def append(self, state, action, reward, next_state, done):
        """Add transition to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def get(self):
        """
        Get n-step transition if buffer is full.
        
        Returns:
            (state_0, action_0, n_step_reward, state_n, done_n) or None
        """
        if len(self.buffer) < self.n_steps:
            return None
        
        # Calculate n-step return
        n_step_reward = 0
        for i, (_, _, reward, _, _) in enumerate(self.buffer):
            n_step_reward += (self.gamma ** i) * reward
        
        # Get first state and action
        state_0, action_0, _, _, _ = self.buffer[0]
        
        # Get final next_state and done
        _, _, _, next_state_n, done_n = self.buffer[-1]
        
        return (state_0, action_0, n_step_reward, next_state_n, done_n)
    
    def reset(self):
        """Clear buffer (called at episode end)"""
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)
