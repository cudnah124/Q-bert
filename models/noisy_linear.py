import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer for exploration via parameter space noise.
    
    Replaces ε-greedy exploration with learnable noise parameters.
    Uses factorized Gaussian noise for efficiency.
    
    Paper: Fortunato et al. 2017 - "Noisy Networks for Exploration"
    https://arxiv.org/abs/1706.10295
    """
    
    def __init__(self, in_features, out_features, sigma_init=0.5):
        """
        Args:
            in_features: Size of input
            out_features: Size of output
            sigma_init: Initial value for sigma (noise standard deviation)
        """
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        # Cache noise tensors to avoid CPU→GPU copies (they'll be moved to device with module)
        self.register_buffer('epsilon_in_cache', torch.zeros(in_features))
        self.register_buffer('epsilon_out_cache', torch.zeros(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters"""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        """Sample new noise for both weight and bias - uses cached device tensors"""
        # Generate noise directly on device (tensors already on correct device via register_buffer)
        self.epsilon_in_cache.normal_()
        self.epsilon_out_cache.normal_()
        
        # Scale noise using factorized Gaussian: sign(x) * sqrt(|x|)
        epsilon_in = self.epsilon_in_cache.sign().mul(self.epsilon_in_cache.abs().sqrt())
        epsilon_out = self.epsilon_out_cache.sign().mul(self.epsilon_out_cache.abs().sqrt())
        
        # Factorized Gaussian noise: outer product of input and output noise
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, x):
        """
        Forward pass with noisy parameters.
        
        weight = μ_w + σ_w * ε_w
        bias = μ_b + σ_b * ε_b
        """
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
