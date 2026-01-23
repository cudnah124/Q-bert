import torch
import torch.nn as nn
import torch.nn.functional as F
from models.noisy_linear import NoisyLinear


class RainbowDQN(nn.Module):
    """
    Rainbow DQN: Combines 6 improvements to DQN.
    
    1. Double Q-Learning (van Hasselt et al. 2015)
    2. Dueling Networks (Wang et al. 2015)
    3. Prioritized Experience Replay (Schaul et al. 2015)
    4. Multi-step Learning (Sutton & Barto 1998)
    5. Distributional RL - C51 (Bellemare et al. 2017)
    6. Noisy Networks (Fortunato et al. 2017)
    
    This network implements Dueling + Noisy Nets + C51 distributional output.
    The other improvements are in the agent/buffer.
    
    Paper: Hessel et al. 2017 - "Rainbow: Combining Improvements in Deep RL"
    https://arxiv.org/abs/1710.02298
    """
    
    def __init__(self, input_channels=4, num_actions=6, num_atoms=51, v_min=-10, v_max=10):
        """
        Args:
            input_channels: Number of stacked frames
            num_actions: Number of discrete actions
            num_atoms: Number of atoms for distributional RL (C51)
            v_min: Minimum value for value distribution support
            v_max: Maximum value for value distribution support
        """
        super(RainbowDQN, self).__init__()
        
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # Support for distributional RL
        self.register_buffer('support', torch.linspace(v_min, v_max, num_atoms))
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        # Convolutional layers (same as DQN Nature 2015)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate conv output size: 84x84 -> 20x20 -> 9x9 -> 7x7
        conv_output_size = 7 * 7 * 64
        
        # Dueling architecture with Noisy Networks
        # Value stream: V(s)
        self.value_fc = NoisyLinear(conv_output_size, 512)
        self.value_out = NoisyLinear(512, num_atoms)
        
        # Advantage stream: A(s,a) for each action
        self.advantage_fc = NoisyLinear(conv_output_size, 512)
        self.advantage_out = NoisyLinear(512, num_actions * num_atoms)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input state [batch, channels, height, width]
        
        Returns:
            Distribution over atoms for each action [batch, num_actions, num_atoms]
            Values are log probabilities (logits before softmax)
        """
        # Normalize input to [0, 1]
        x = x / 255.0
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        
        # Dueling streams
        # Value stream: [batch, num_atoms]
        value = F.relu(self.value_fc(x))
        value = self.value_out(value).view(-1, 1, self.num_atoms)
        
        # Advantage stream: [batch, num_actions, num_atoms]
        advantage = F.relu(self.advantage_fc(x))
        advantage = self.advantage_out(advantage).view(-1, self.num_actions, self.num_atoms)
        
        # Combine: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        # This ensures identifiability of V and A
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Apply softmax over atoms to get probability distribution
        # Return log probabilities for numerical stability
        return F.log_softmax(q_atoms, dim=2)
    
    def reset_noise(self):
        """Reset noise in all noisy layers"""
        self.value_fc.reset_noise()
        self.value_out.reset_noise()
        self.advantage_fc.reset_noise()
        self.advantage_out.reset_noise()
    
    def get_q_values(self, x):
        """
        Get expected Q-values (for action selection).
        
        Q(s,a) = Σ_i z_i * p_i(s,a)
        where z_i are support atoms and p_i are probabilities
        """
        log_probs = self.forward(x)
        probs = log_probs.exp()
        q_values = (probs * self.support.view(1, 1, -1)).sum(dim=2)
        return q_values
