import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class RainbowDQNAgent:
    """
    Rainbow DQN Agent - Combines 6 improvements to DQN.
    
    1. Double Q-Learning: Decouples action selection and evaluation
    2. Dueling Architecture: Separate value and advantage streams (in model)
    3. Prioritized Replay: Samples important transitions (uses PrioritizedReplayBuffer)
    4. Multi-step Learning: Uses n-step returns
    5. Distributional RL (C51): Models value distribution instead of expected value
    6. Noisy Networks: Parameter space exploration (in model)
    
    Paper: Hessel et al. 2017 - "Rainbow: Combining Improvements in Deep RL"
    https://arxiv.org/abs/1710.02298
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
        
        # C51 parameters
        self.num_atoms = config.NUM_ATOMS
        self.v_min = config.V_MIN
        self.v_max = config.V_MAX
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(self.device)
        
        # Cache offset tensor for categorical projection (batch_size is constant)
        self.offset = torch.linspace(
            0, (config.BATCH_SIZE - 1) * self.num_atoms, config.BATCH_SIZE
        ).long().unsqueeze(1).expand(config.BATCH_SIZE, self.num_atoms).to(self.device)
    
    def select_action(self, state):
        """
        Select action using noisy networks (no epsilon needed).
        
        Noisy networks provide exploration through parameter noise,
        so we always select the greedy action.
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device) / 255.0
            q_values = self.policy_net.get_q_values(state_tensor)
            return q_values.argmax(1).item()
    
    def get_max_q_value(self, state):
        """Get maximum Q-value for state (for logging)"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device) / 255.0
            q_values = self.policy_net.get_q_values(state_tensor)
            return q_values.max().item()
    
    def train_step(self, replay_buffer):
        """
        Training step with distributional loss.
        
        Uses categorical cross-entropy loss between current and target distributions.
        The target distribution is projected onto the support using the Bellman update.
        """
        # Sample from prioritized buffer
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
        
        # Current distribution: log p(s, a)
        log_probs = self.policy_net(states)
        actions_expanded = actions.unsqueeze(1).unsqueeze(1).expand(-1, 1, self.num_atoms)
        log_probs = log_probs.gather(1, actions_expanded).squeeze(1)
        
        # Target distribution using Double Q-Learning + distributional Bellman
        with torch.no_grad():
            # Double Q-Learning: use online network to select best actions
            next_q_values = self.policy_net.get_q_values(next_states)
            next_actions = next_q_values.argmax(1)
            
            # Get target network's distribution for selected actions
            next_log_probs = self.target_net(next_states)
            next_actions_expanded = next_actions.unsqueeze(1).unsqueeze(1).expand(-1, 1, self.num_atoms)
            next_log_probs = next_log_probs.gather(1, next_actions_expanded).squeeze(1)
            next_probs = next_log_probs.exp()
            
            # Distributional Bellman update: project onto support
            target_distribution = self._categorical_projection(
                rewards, dones, next_probs
            )
        
        # Cross-entropy loss between current and target distributions
        elementwise_loss = -(target_distribution * log_probs).sum(dim=1)
        
        # Apply importance sampling weights
        loss = (elementwise_loss * weights).mean()
        
        # Calculate TD errors for priority updates (use KL divergence)
        with torch.no_grad():
            td_errors = (target_distribution * (target_distribution.log() - log_probs)).sum(dim=1)
            td_errors = td_errors.cpu().numpy()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        
        # Reset noise in noisy layers
        self.policy_net.reset_noise()
        self.target_net.reset_noise()
        
        # Update priorities
        replay_buffer.update_priorities(indices, td_errors)
        
        return loss.item()
    
    def _categorical_projection(self, rewards, dones, next_probs):
        """
        Categorical projection for distributional Bellman update.
        
        Projects the target distribution T_z = r + γz onto the support.
        
        Args:
            rewards: [batch_size]
            dones: [batch_size]
            next_probs: [batch_size, num_atoms]
        
        Returns:
            target_distribution: [batch_size, num_atoms]
        """
        batch_size = rewards.size(0)
        
        # Compute projected value for each atom
        # T_z = r + γ * (1 - done) * z
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)
        support = self.support.unsqueeze(0)
        
        T_z = rewards + self.config.GAMMA * (1 - dones) * support
        T_z = T_z.clamp(self.v_min, self.v_max)
        
        # Compute projection indices
        b = (T_z - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()
        
        # Fix edge cases
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.num_atoms - 1)) * (l == u)] += 1
        
        # Distribute probability
        target_distribution = torch.zeros_like(next_probs)
        
        # Use cached offset tensor (already on device)
        target_distribution.view(-1).index_add_(
            0, (l + self.offset).view(-1), (next_probs * (u.float() - b)).view(-1)
        )
        target_distribution.view(-1).index_add_(
            0, (u + self.offset).view(-1), (next_probs * (b - l.float())).view(-1)
        )
        
        return target_distribution
    
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
