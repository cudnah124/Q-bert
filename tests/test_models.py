import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.vanilla_dqn import VanillaDQN
from models.double_dqn import DoubleDQN
from models.dueling_dqn import DuelingDQN

def test_vanilla_dqn():
    print("Testing VanillaDQN...")
    
    model = VanillaDQN(input_channels=4, num_actions=6)
    dummy_input = torch.randn(2, 4, 84, 84)
    output = model(dummy_input)
    
    assert output.shape == (2, 6), f"Wrong output shape: {output.shape}"
    print(f"✓ VanillaDQN output shape: {output.shape}")

def test_double_dqn():
    print("Testing DoubleDQN...")
    
    model = DoubleDQN(input_channels=4, num_actions=6)
    dummy_input = torch.randn(2, 4, 84, 84)
    output = model(dummy_input)
    
    assert output.shape == (2, 6), f"Wrong output shape: {output.shape}"
    print(f"✓ DoubleDQN output shape: {output.shape}")

def test_dueling_dqn():
    print("Testing DuelingDQN...")
    
    model = DuelingDQN(input_channels=4, num_actions=6)
    dummy_input = torch.randn(2, 4, 84, 84)
    output = model(dummy_input)
    
    assert output.shape == (2, 6), f"Wrong output shape: {output.shape}"
    print(f"✓ DuelingDQN output shape: {output.shape}")

def test_parameter_count():
    print("\nParameter counts:")
    
    vanilla = VanillaDQN()
    double = DoubleDQN()
    dueling = DuelingDQN()
    
    vanilla_params = sum(p.numel() for p in vanilla.parameters())
    double_params = sum(p.numel() for p in double.parameters())
    dueling_params = sum(p.numel() for p in dueling.parameters())
    
    print(f"  VanillaDQN: {vanilla_params:,} parameters")
    print(f"  DoubleDQN:  {double_params:,} parameters")
    print(f"  DuelingDQN: {dueling_params:,} parameters")

if __name__ == "__main__":
    test_vanilla_dqn()
    test_double_dqn()
    test_dueling_dqn()
    test_parameter_count()
    print("\n✅ All model tests passed!")
