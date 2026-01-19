import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tempfile
import shutil
from utils.logger import DQNLogger

def test_logger():
    print("Testing DQNLogger...")
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        logger = DQNLogger(temp_dir, "test_algo")
        
        # Test config save
        logger.save_config({'learning_rate': 0.0001, 'gamma': 0.99})
        assert os.path.exists(os.path.join(temp_dir, "config.json"))
        print("✓ Config saved")
        
        # Test step logging
        logger.log_step(
            global_step=10,
            episode=1,
            step_in_episode=10,
            action=2,
            reward=1.0,
            loss=0.5,
            q_value=2.5,
            epsilon=0.9
        )
        print("✓ Step logged")
        
        # Test episode logging
        logger.log_episode(
            episode=1,
            total_reward=100,
            episode_length=50,
            epsilon=0.9,
            total_steps=50,
            training_time=10.5,
            level_reached=1,
            buffer_reward_stats={'min': -1, 'max': 25, 'mean': 2.5, 'std': 5.0}
        )
        print("✓ Episode logged")
        
        logger.close()
        
        # Check files exist
        assert os.path.exists(os.path.join(temp_dir, "training_log.csv"))
        assert os.path.exists(os.path.join(temp_dir, "step_log.csv"))
        print("✓ Log files created")
        
        # Check CSV content
        with open(os.path.join(temp_dir, "training_log.csv"), 'r') as f:
            lines = f.readlines()
            assert len(lines) == 2  # Header + 1 episode
            assert 'training_time_seconds' in lines[0]
            assert 'level_reached' in lines[0]
            assert 'buffer_reward_mean' in lines[0]
        print("✓ CSV format correct")
        
    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    test_logger()
    print("\n✅ All logger tests passed!")
