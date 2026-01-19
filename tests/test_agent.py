import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from models.vanilla_dqn import VanillaDQN
from agents.vanilla_dqn_agent import VanillaDQNAgent
from utils.replay_buffer import ReplayBuffer
import config.vanilla_dqn_config as config

def test_agent_training():
    print("Testing agent training loop...")
    
    # Setup
    policy_net = VanillaDQN()
    target_net = VanillaDQN()
    agent = VanillaDQNAgent(policy_net, target_net, config)
    
    replay_buffer = ReplayBuffer(capacity=1000)
    
    # Fill buffer with dummy data
    print("Filling replay buffer...")
    for i in range(100):
        state = np.random.randn(4, 84, 84)
        action = np.random.randint(0, 6)
        reward = np.random.randn()
        next_state = np.random.randn(4, 84, 84)
        done = False
        replay_buffer.push(state, action, reward, next_state, done)
    
    print(f"✓ Buffer filled: {len(replay_buffer)} transitions")
    
    # Test action selection
    state = np.random.randn(4, 84, 84)
    action = agent.select_action(state, epsilon=0.5)
    assert 0 <= action < 6, f"Invalid action: {action}"
    print(f"✓ Action selection works: {action}")
    
    # Test Q-value
    q_value = agent.get_max_q_value(state)
    assert isinstance(q_value, float)
    print(f"✓ Q-value: {q_value:.3f}")
    
    # Test training step
    loss = agent.train_step(replay_buffer)
    assert isinstance(loss, float) and loss >= 0
    print(f"✓ Training step works, loss: {loss:.4f}")
    
    # Test target network update
    old_params = [p.clone() for p in agent.target_net.parameters()]
    agent.update_target_network()
    new_params = list(agent.target_net.parameters())
    
    # Check if params changed
    changed = any(not torch.equal(old, new) for old, new in zip(old_params, new_params))
    assert changed, "Target network not updated"
    print("✓ Target network update works")

if __name__ == "__main__":
    test_agent_training()
    print("\n✅ All agent tests passed!")
