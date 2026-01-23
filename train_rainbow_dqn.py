import gymnasium as gym
import os
import time
from tqdm import tqdm

from config.rainbow_config import *
from models.rainbow_dqn import RainbowDQN
from agents.rainbow_dqn_agent import RainbowDQNAgent
from utils.preprocessing import preprocess_frame, FrameStack
from utils.prioritized_replay_buffer import PrioritizedReplayBuffer
from utils.multi_step_buffer import MultiStepBuffer
from utils.logger import DQNLogger
from utils.qbert_utils import get_level_from_ram

# Directories
LOG_DIR = "logs/rainbow_dqn"
CHECKPOINT_DIR = "checkpoints/rainbow_dqn"
MIN_REPLAY_SIZE = TRAIN_START
TOTAL_EPISODES = MAX_EPISODES
MAX_STEPS_PER_EPISODE = 10000
STEP_LOG_FREQUENCY = 1000

VERBOSE = False

def train_rainbow_dqn():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    import ale_py
    gym.register_envs(ale_py)
    
    env = gym.make("ALE/Qbert-v5")
    
    policy_net = RainbowDQN(
        input_channels=FRAME_STACK,
        num_actions=NUM_ACTIONS,
        num_atoms=NUM_ATOMS,
        v_min=V_MIN,
        v_max=V_MAX
    )
    target_net = RainbowDQN(
        input_channels=FRAME_STACK,
        num_actions=NUM_ACTIONS,
        num_atoms=NUM_ATOMS,
        v_min=V_MIN,
        v_max=V_MAX
    )
    
    import config.rainbow_config as config
    agent = RainbowDQNAgent(policy_net, target_net, config)
    
    # Use Prioritized Replay Buffer
    replay_buffer = PrioritizedReplayBuffer(
        capacity=REPLAY_BUFFER_SIZE,
        alpha=ALPHA,
        beta_start=BETA_START,
        beta_frames=BETA_FRAMES,
        epsilon=PRIORITY_EPSILON
    )
    
    # Multi-step buffer for n-step returns
    multistep_buffer = MultiStepBuffer(n_steps=N_STEPS, gamma=GAMMA)
    
    logger = DQNLogger(LOG_DIR, "rainbow_dqn", enabled=False)
    logger.save_config({
        'algorithm': 'Rainbow DQN',
        'learning_rate': LEARNING_RATE,
        'gamma': GAMMA,
        'batch_size': BATCH_SIZE,
        'replay_buffer_size': REPLAY_BUFFER_SIZE,
        'target_update': TARGET_UPDATE,
        'alpha': ALPHA,
        'beta_start': BETA_START,
        'beta_frames': BETA_FRAMES,
        'n_steps': N_STEPS,
        'num_atoms': NUM_ATOMS,
        'v_min': V_MIN,
        'v_max': V_MAX,
        'sigma_init': SIGMA_INIT,
    })
    
    global_step = 0
    # No epsilon for Rainbow - uses Noisy Networks for exploration
    
    for episode in tqdm(range(TOTAL_EPISODES), desc="Training Rainbow DQN"):
        episode_start_time = time.time()
        
        obs, info = env.reset()
        frame_stack = FrameStack(num_frames=FRAME_STACK)
        state = frame_stack.reset(obs)
        
        episode_reward = 0
        episode_length = 0
        level_reached = 1
        
        # Reset multi-step buffer at episode start
        multistep_buffer.reset()
        
        for step in range(MAX_STEPS_PER_EPISODE):
            # Noisy networks - no epsilon needed
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Clip rewards to [-1, 1]
            reward = max(-1.0, min(1.0, reward))
            
            done = terminated or truncated
            next_state = frame_stack.step(next_obs)
            
            # Extract level from RAM
            current_level = get_level_from_ram(env.unwrapped.ale)
            level_reached = max(level_reached, current_level)
            
            # Add to multi-step buffer
            multistep_buffer.append(state, action, reward, next_state, done)
            
            # Get n-step transition if available
            n_step_transition = multistep_buffer.get()
            if n_step_transition is not None:
                s0, a0, n_step_reward, sn, done_n = n_step_transition
                replay_buffer.push(s0, a0, n_step_reward, sn, done_n)
            
            loss = None
            if len(replay_buffer) > MIN_REPLAY_SIZE and global_step % UPDATE_FREQUENCY == 0:
                loss = agent.train_step(replay_buffer)
            
            if global_step % TARGET_UPDATE == 0:
                agent.update_target_network()
            
            if global_step % STEP_LOG_FREQUENCY == 0:
                q_value = agent.get_max_q_value(state)
                # Log epsilon as 0.0 for Rainbow (uses noisy nets)
                logger.log_step(global_step, episode, step, action, reward, loss, q_value, 0.0)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            global_step += 1
            
            if done:
                # Reset multi-step buffer on episode end
                multistep_buffer.reset()
                break
        
        episode_time = time.time() - episode_start_time
        buffer_stats = replay_buffer.get_reward_stats()
        
        # Log episode (epsilon = 0.0 for Rainbow)
        logger.log_episode(episode, episode_reward, episode_length, 0.0, global_step,
                          episode_time, level_reached, buffer_stats)
        
        if (episode + 1) % SAVE_FREQUENCY == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"ep_{episode+1}.pth")
            agent.save(checkpoint_path)
            if VERBOSE:
                print(f"\nSaved: {checkpoint_path}")
        
        if (episode + 1) % 100 == 0:
            if VERBOSE:
                print(f"\nEp {episode+1}/{TOTAL_EPISODES} | Reward: {episode_reward:.0f} | "
                    f"Steps: {episode_length}")
    
    final_path = os.path.join(CHECKPOINT_DIR, "final.pth")
    agent.save(final_path)
    logger.close()
    env.close()
    if VERBOSE:
        print(f"\nDone! Saved to {final_path}")

if __name__ == "__main__":
    train_rainbow_dqn()
