import numpy as np
import random


class SumTree:
    """
    Sum Tree data structure for efficient proportional prioritization sampling.
    
    Binary tree where:
    - Leaf nodes store priorities
    - Parent nodes store sum of children
    - Root stores total sum of all priorities
    
    This allows O(log n) sampling and updates.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0
    
    def _propagate(self, idx, change):
        """Update parent nodes after leaf change"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        """Find leaf node given cumulative sum s"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        """Total sum of all priorities"""
        return self.tree[0]
    
    def add(self, priority, data):
        """Add new experience with priority"""
        idx = self.write + self.capacity - 1
        
        self.data[self.write] = data
        self.update(idx, priority)
        
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx, priority):
        """Update priority of node at idx"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s):
        """Sample experience proportional to priority"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    
    Samples transitions based on their TD error priorities using a sum tree.
    Implements importance sampling to correct for non-uniform sampling bias.
    
    Paper: Schaul et al. 2015 - "Prioritized Experience Replay"
    https://arxiv.org/abs/1511.05952
    """
    
    def __init__(self, capacity=100000, alpha=0.6, beta_start=0.4, beta_frames=100000, epsilon=0.01):
        """
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0=uniform, 1=fully prioritized)
            beta_start: Initial importance sampling weight
            beta_frames: Number of frames to anneal beta to 1.0
            epsilon: Small constant added to priorities to ensure non-zero sampling
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        # Track max priority for new experiences
        self.max_priority = 1.0
    
    def _get_priority(self, error):
        """Convert TD error to priority: p = (|error| + ε)^α"""
        return (np.abs(error) + self.epsilon) ** self.alpha
    
    def push(self, state, action, reward, next_state, done):
        """Add experience with maximum priority (will be updated after training)"""
        data = (state.astype(np.float32), action, np.float32(reward), 
                next_state.astype(np.float32), np.float32(done))
        self.tree.add(self.max_priority, data)
    
    def sample(self, batch_size=32):
        """
        Sample batch with priorities.
        
        Returns:
            batch: (states, actions, rewards, next_states, dones)
            indices: Tree indices for updating priorities
            weights: Importance sampling weights
        """
        batch = []
        indices = []
        priorities = []
        
        # Current beta value (annealed from beta_start to 1.0)
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        
        # Divide priority range into batch_size segments
        segment = self.tree.total() / batch_size
        
        for i in range(batch_size):
            # Sample uniformly from each segment
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, priority, data = self.tree.get(s)
            
            if data is not None:
                batch.append(data)
                indices.append(idx)
                priorities.append(priority)
        
        # Calculate importance sampling weights
        # w = (N * P(i))^(-β) / max_w
        sampling_probabilities = np.array(priorities) / self.tree.total()
        weights = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        weights /= weights.max()  # Normalize by max for stability
        
        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        self.frame += 1
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32)
        )
    
    def update_priorities(self, indices, errors):
        """
        Update priorities based on TD errors.
        
        Args:
            indices: Tree indices from sample()
            errors: TD errors (|target - prediction|)
        """
        for idx, error in zip(indices, errors):
            priority = self._get_priority(error)
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def get_reward_stats(self):
        """Get statistics about rewards in buffer (for logging)"""
        if self.tree.n_entries == 0:
            return {'min': 0, 'max': 0, 'mean': 0, 'std': 0}
        
        rewards = []
        for i in range(self.tree.n_entries):
            data = self.tree.data[i]
            if data is not None:
                rewards.append(data[2])
        
        return {
            'min': np.min(rewards),
            'max': np.max(rewards),
            'mean': np.mean(rewards),
            'std': np.std(rewards)
        }
    
    def __len__(self):
        return self.tree.n_entries
