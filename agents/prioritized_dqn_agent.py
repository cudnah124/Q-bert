import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np


class PrioritizedDQNAgent:
    """
    Prioritized DQN Agent.
    
    Uses prioritized experience replay to sample important transitions
    more frequently, leading to faster learning.
    
    Key differences from Vanilla DQN:
    - Uses PrioritizedReplayBuffer instead of standard ReplayBuffer
    - Applies importance sampling weights to loss
    - Updates priorities after each training step
    
    Paper: Schaul et al. 2015 - "Prioritized Experience Replay"
    """
    
    def __init__(self, policy_net, target_net, config):
        self.policy_net = policy_net
        self.target_net = target_net
        self.config = config
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
    
    def select_action(self, state, epsilon):
        """Select action using epsilon-greedy policy"""
        if random.random() < epsilon:
            return random.randrange(self.config.NUM_ACTIONS)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(1).item()
    
    def get_max_q_value(self, state):
        """Get maximum Q-value for state (for logging)"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device) / 255.0
            q_values = self.policy_net(state_tensor)
            return q_values.max().item()
    
    def train_step(self, replay_buffer):
        """
        Training step with prioritized replay.
        
        Samples from prioritized buffer, computes weighted loss,
        and updates priorities based on TD errors.
        """
        # Sample from prioritized buffer (returns indices and weights)
        states, actions, rewards, next_states, dones, indices, weights = \
            replay_buffer.sample(self.config.BATCH_SIZE)
        
        # CRITICAL FIX: Normalize uint8 states [0-255] to float [0-1]
        states = torch.FloatTensor(states).to(self.device) / 255.0
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        # CRITICAL FIX: Normalize uint8 states [0-255] to float [0-1]
        next_states = torch.FloatTensor(next_states).to(self.device) / 255.0
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Current Q-values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values (using target network)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.config.GAMMA * next_q * (1 - dones)
        
        # Calculate TD errors for priority updates
        td_errors = torch.abs(current_q - target_q).detach().cpu().numpy()
        
        # Weighted Huber loss (importance sampling)
        elementwise_loss = F.smooth_l1_loss(current_q, target_q, reduction='none')
        loss = (elementwise_loss * weights).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        
        # Update priorities in replay buffer
        replay_buffer.update_priorities(indices, td_errors)
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights from policy network to target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path):
        """Save model checkpoint"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
