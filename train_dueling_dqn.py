import gymnasium as gym
import os
import time
from tqdm import tqdm

from config.dueling_dqn_config import *
from models.dueling_dqn import DuelingDQN
from agents.dueling_dqn_agent import DuelingDQNAgent
from utils.preprocessing import preprocess_frame, FrameStack
from utils.replay_buffer import ReplayBuffer
from utils.logger import DQNLogger

def train_dueling_dqn():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    import ale_py
    gym.register_envs(ale_py)
    
    env = gym.make("ALE/Qbert-v5")
    
    policy_net = DuelingDQN(input_channels=FRAME_STACK, num_actions=NUM_ACTIONS)
    target_net = DuelingDQN(input_channels=FRAME_STACK, num_actions=NUM_ACTIONS)
    
    import config.dueling_dqn_config as config
    agent = DuelingDQNAgent(policy_net, target_net, config)
    
    replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_SIZE)
    
    logger = DQNLogger(LOG_DIR, "dueling_dqn")
    logger.save_config({
        'algorithm': 'Dueling DQN',
        'learning_rate': LEARNING_RATE,
        'gamma': GAMMA,
        'batch_size': BATCH_SIZE,
        'epsilon_start': EPSILON_START,
        'epsilon_end': EPSILON_END,
        'epsilon_decay_steps': EPSILON_DECAY_STEPS,
        'replay_buffer_size': REPLAY_BUFFER_SIZE,
        'target_update': TARGET_UPDATE,
    })
    
    global_step = 0
    epsilon = EPSILON_START
    epsilon_decay = (EPSILON_START - EPSILON_END) / EPSILON_DECAY_STEPS
    
    for episode in tqdm(range(TOTAL_EPISODES), desc="Training Dueling DQN"):
        episode_start_time = time.time()
        
        obs, info = env.reset()
        frame_stack = FrameStack(num_frames=FRAME_STACK)
        state = frame_stack.reset(obs)
        
        episode_reward = 0
        episode_length = 0
        level_reached = 1
        
        for step in range(MAX_STEPS_PER_EPISODE):
            action = agent.select_action(state, epsilon)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = frame_stack.step(next_obs)
            
            if 'level' in info:
                level_reached = max(level_reached, info['level'])
            
            replay_buffer.push(state, action, reward, next_state, done)
            
            loss = None
            if len(replay_buffer) > MIN_REPLAY_SIZE and global_step % UPDATE_FREQUENCY == 0:
                loss = agent.train_step(replay_buffer)
            
            if global_step % TARGET_UPDATE == 0:
                agent.update_target_network()
            
            if global_step % STEP_LOG_FREQUENCY == 0:
                q_value = agent.get_max_q_value(state)
                logger.log_step(global_step, episode, step, action, reward, loss, q_value, epsilon)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            global_step += 1
            epsilon = max(EPSILON_END, epsilon - epsilon_decay)
            
            if done:
                break
        
        episode_time = time.time() - episode_start_time
        buffer_stats = replay_buffer.get_reward_stats()
        
        logger.log_episode(episode, episode_reward, episode_length, epsilon, global_step,
                          episode_time, level_reached, buffer_stats)
        
        if (episode + 1) % SAVE_FREQUENCY == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"ep_{episode+1}.pth")
            agent.save(checkpoint_path)
            print(f"\nSaved: {checkpoint_path}")
        
        if (episode + 1) % 100 == 0:
            print(f"\nEp {episode+1}/{TOTAL_EPISODES} | Reward: {episode_reward:.0f} | "
                  f"Steps: {episode_length} | Eps: {epsilon:.3f}")
    
    final_path = os.path.join(CHECKPOINT_DIR, "final.pth")
    agent.save(final_path)
    logger.close()
    env.close()
    print(f"\nDone! Saved to {final_path}")

if __name__ == "__main__":
    train_dueling_dqn()
