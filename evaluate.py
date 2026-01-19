import gymnasium as gym
import torch
import numpy as np
import argparse
from utils.preprocessing import preprocess_frame, FrameStack

def evaluate(checkpoint_path, algorithm, num_episodes=10, render=False):
    if algorithm == "vanilla_dqn":
        from models.vanilla_dqn import VanillaDQN
        model = VanillaDQN()
    elif algorithm == "double_dqn":
        from models.double_dqn import DoubleDQN
        model = DoubleDQN()
    elif algorithm == "dueling_dqn":
        from models.dueling_dqn import DuelingDQN
        model = DuelingDQN()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['policy_net_state_dict'])
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    import ale_py
    gym.register_envs(ale_py)
    
    env = gym.make("ALE/Qbert-v5")
    scores = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        frame_stack = FrameStack(num_frames=4)
        state = frame_stack.reset(obs)
        
        episode_reward = 0
        
        while True:
            if render:
                env.render()
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                action = q_values.argmax(1).item()
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = frame_stack.step(next_obs)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        scores.append(episode_reward)
        print(f"Episode {episode+1}/{num_episodes}: {episode_reward:.0f}")
    
    env.close()
    
    print(f"\n{'='*50}")
    print(f"Algorithm: {algorithm}")
    print(f"Average: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Min: {np.min(scores):.0f} | Max: {np.max(scores):.0f}")
    print(f"{'='*50}")
    
    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--algorithm", type=str, required=True, 
                       choices=["vanilla_dqn", "double_dqn", "dueling_dqn"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    
    args = parser.parse_args()
    evaluate(args.checkpoint, args.algorithm, args.episodes, args.render)
