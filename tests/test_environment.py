import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym

def test_environment():
    print("Testing Gymnasium environment...")
    
    try:
        # Register ALE environments
        import ale_py
        gym.register_envs(ale_py)
        
        env = gym.make("ALE/Qbert-v5")
        print("✓ Environment created")
        
        obs, info = env.reset()
        assert obs.shape == (210, 160, 3), f"Wrong observation shape: {obs.shape}"
        print(f"✓ Observation shape: {obs.shape}")
        
        assert env.action_space.n == 6, f"Wrong action space: {env.action_space.n}"
        print(f"✓ Action space: {env.action_space.n}")
        
        # Test step
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        assert next_obs.shape == (210, 160, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        print("✓ Step works")
        
        env.close()
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        print("\nTo fix:")
        print("  pip install gymnasium[atari]")
        print("  pip install shimmy[gym-v26]")
        print("  pip install autorom[accept-rom-license]")
        print("  AutoROM --accept-license")
        raise

if __name__ == "__main__":
    test_environment()
    print("\n✅ Environment test passed!")
