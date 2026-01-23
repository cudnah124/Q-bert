import torch
import torch.nn.functional as F
import torch.optim as optim
import random

class DoubleDQNAgent:
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
        if random.random() < epsilon:
            return random.randrange(self.config.NUM_ACTIONS)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device) / 255.0
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(1).item()
    
    def get_max_q_value(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device) / 255.0
            q_values = self.policy_net(state_tensor)
            return q_values.max().item()
    
    def train_step(self, replay_buffer):
        states, actions, rewards, next_states, dones = replay_buffer.sample(self.config.BATCH_SIZE)
        
        # CRITICAL FIX: Normalize uint8 states [0-255] to float [0-1]
        states = torch.FloatTensor(states).to(self.device) / 255.0
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        # CRITICAL FIX: Normalize uint8 states [0-255] to float [0-1]
        next_states = torch.FloatTensor(next_states).to(self.device) / 255.0
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # double dqn selection
        with torch.no_grad():
            best_actions = self.policy_net(next_states).argmax(1)
            next_q = self.target_net(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.config.GAMMA * next_q * (1 - dones)
        
        # Use Huber loss (smooth_l1_loss) as per DQN Nature 2015 paper
        loss = F.smooth_l1_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
