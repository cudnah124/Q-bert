import csv
import json
import os
from datetime import datetime

class DQNLogger:
    """Logging system for DQN training"""
    
    def __init__(self, log_dir, algorithm_name):
        self.log_dir = log_dir
        self.algorithm_name = algorithm_name
        os.makedirs(log_dir, exist_ok=True)
        
        # episode
        self.episode_log_path = os.path.join(log_dir, "training_log.csv")
        self.episode_log_file = open(self.episode_log_path, 'w', newline='')
        self.episode_writer = csv.writer(self.episode_log_file)
        self.episode_writer.writerow([
            'episode', 'total_reward', 'episode_length', 'avg_loss', 'avg_q_value',
            'epsilon', 'timestamp', 'total_steps', 'training_time_seconds',
            'level_reached', 'buffer_reward_min', 'buffer_reward_max',
            'buffer_reward_mean', 'buffer_reward_std'
        ])
        
        # step
        self.step_log_path = os.path.join(log_dir, "step_log.csv")
        self.step_log_file = open(self.step_log_path, 'w', newline='')
        self.step_writer = csv.writer(self.step_log_file)
        self.step_writer.writerow([
            'global_step', 'episode', 'step_in_episode', 'action',
            'reward', 'loss', 'q_value', 'epsilon', 'timestamp'
        ])
        
        self.episode_metrics = {'losses': [], 'q_values': [], 'rewards': []}
    
    def log_step(self, global_step, episode, step_in_episode, action, 
                 reward, loss, q_value, epsilon):
        self.step_writer.writerow([
            global_step, episode, step_in_episode, action, reward,
            loss if loss is not None else '', q_value, epsilon,
            datetime.now().isoformat()
        ])
        
        self.episode_metrics['rewards'].append(reward)
        if loss is not None:
            self.episode_metrics['losses'].append(loss)
        if q_value is not None:
            self.episode_metrics['q_values'].append(q_value)
    
    def log_episode(self, episode, total_reward, episode_length, epsilon,
                    total_steps, training_time, level_reached=1,
                    buffer_reward_stats=None):
        avg_loss = sum(self.episode_metrics['losses']) / len(self.episode_metrics['losses']) \
                   if self.episode_metrics['losses'] else 0
        avg_q = sum(self.episode_metrics['q_values']) / len(self.episode_metrics['q_values']) \
                if self.episode_metrics['q_values'] else 0
        
        if buffer_reward_stats is None:
            buffer_reward_stats = {'min': 0, 'max': 0, 'mean': 0, 'std': 0}
        
        self.episode_writer.writerow([
            episode, total_reward, episode_length, avg_loss, avg_q, epsilon,
            datetime.now().isoformat(), total_steps, training_time, level_reached,
            buffer_reward_stats['min'], buffer_reward_stats['max'],
            buffer_reward_stats['mean'], buffer_reward_stats['std']
        ])
        
        self.episode_log_file.flush()
        self.step_log_file.flush()
        
        self.episode_metrics = {'losses': [], 'q_values': [], 'rewards': []}
    
    def save_config(self, config_dict):
        config_path = os.path.join(self.log_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def close(self):
        self.episode_log_file.close()
        self.step_log_file.close()
