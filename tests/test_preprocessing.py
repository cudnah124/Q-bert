import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from utils.preprocessing import preprocess_frame, FrameStack

def test_preprocess_frame():
    print("Testing preprocess_frame...")
    
    # Create dummy frame
    frame = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)
    processed = preprocess_frame(frame)
    
    assert processed.shape == (84, 84), f"Wrong shape: {processed.shape}"
    assert 0 <= processed.min() <= processed.max() <= 1, "Not normalized"
    
    print("✓ preprocess_frame works")

def test_frame_stack():
    print("Testing FrameStack...")
    
    frame_stack = FrameStack(num_frames=4)
    
    # Test reset
    dummy_frame = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)
    state = frame_stack.reset(dummy_frame)
    
    assert state.shape == (4, 84, 84), f"Wrong shape: {state.shape}"
    
    # Test step
    next_frame = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)
    next_state = frame_stack.step(next_frame)
    
    assert next_state.shape == (4, 84, 84), f"Wrong shape: {next_state.shape}"
    
    print("✓ FrameStack works")

if __name__ == "__main__":
    test_preprocess_frame()
    test_frame_stack()
    print("\n✅ All preprocessing tests passed!")
