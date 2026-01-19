import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    def __init__(self, input_channels=4, num_actions=6):
        super(DuelingDQN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # value
        self.value_fc1 = nn.Linear(7 * 7 * 64, 512)
        self.value_fc2 = nn.Linear(512, 1)
        
        # advantage
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
        
        # q(s,a) = v(s) + (a(s,a) - mean(a(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
