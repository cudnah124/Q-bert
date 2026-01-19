import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    def __init__(self, input_channels=4, num_actions=6):
        super(DuelingDQN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Value stream
        self.value_fc1 = nn.Linear(7 * 7 * 64, 512)
        self.value_fc2 = nn.Linear(512, 1)
        
        # Advantage stream
        self.advantage_fc1 = nn.Linear(7 * 7 * 64, 512)
        self.advantage_fc2 = nn.Linear(512, num_actions)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        value = F.relu(self.value_fc1(x))
        value = self.value_fc2(value)
        
        advantage = F.relu(self.advantage_fc1(x))
        advantage = self.advantage_fc2(advantage)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
