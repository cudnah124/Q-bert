import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from utils.replay_buffer import ReplayBuffer

def test_replay_buffer():
    print("Testing ReplayBuffer...")
    
    buffer = ReplayBuffer(capacity=100)
    
    # Test push
    for i in range(50):
        state = np.random.randn(4, 84, 84)
        action = np.random.randint(0, 6)
        reward = np.random.randn()
        next_state = np.random.randn(4, 84, 84)
        done = False
        buffer.push(state, action, reward, next_state, done)
    
    assert len(buffer) == 50, f"Wrong buffer size: {len(buffer)}"
    print(f"✓ Buffer size: {len(buffer)}")
    
    # Test sample
    states, actions, rewards, next_states, dones = buffer.sample(batch_size=32)
    
    assert states.shape == (32, 4, 84, 84), f"Wrong states shape: {states.shape}"
    assert actions.shape == (32,), f"Wrong actions shape: {actions.shape}"
    assert rewards.shape == (32,), f"Wrong rewards shape: {rewards.shape}"
    assert next_states.shape == (32, 4, 84, 84), f"Wrong next_states shape: {next_states.shape}"
    assert dones.shape == (32,), f"Wrong dones shape: {dones.shape}"
    
    print("✓ Sampling works")
    
    # Test reward stats
    stats = buffer.get_reward_stats()
    assert 'min' in stats and 'max' in stats and 'mean' in stats and 'std' in stats
    print(f"✓ Reward stats: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}")

if __name__ == "__main__":
    test_replay_buffer()
    print("\n✅ All replay buffer tests passed!")
