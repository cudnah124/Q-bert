import random
import numpy as np

class ReplayBuffer:
    """Circular buffer for experience replay"""
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        # Store with explicit float32 dtype to save memory
        self.buffer[self.position] = (
            state.astype(np.float32),
            action,
            np.float32(reward),
            next_state.astype(np.float32),
            np.float32(done)
        )
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size=32):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def get_reward_stats(self):
        if len(self.buffer) == 0:
            return {'min': 0, 'max': 0, 'mean': 0, 'std': 0}
        
        rewards = [t[2] for t in self.buffer]
        return {
            'min': np.min(rewards),
            'max': np.max(rewards),
            'mean': np.mean(rewards),
            'std': np.std(rewards)
        }
    
    def __len__(self):
        return len(self.buffer)
