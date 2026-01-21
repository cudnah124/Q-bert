"""
Quick analysis script to check if DQN training is progressing normally
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_training_progress(log_path="logs/vanilla_dqn/training_log.csv"):
    """Analyze training metrics to check learning progress"""
    
    if not os.path.exists(log_path):
        print(f"❌ Log file not found: {log_path}")
        return None
    
    # Detect model type from path
    model_type = "UNKNOWN"
    if "vanilla_dqn" in log_path.lower():
        model_type = "VANILLA DQN"
    elif "double_dqn" in log_path.lower():
        model_type = "DOUBLE DQN"
    elif "dueling_dqn" in log_path.lower():
        model_type = "DUELING DQN"
    
    df = pd.read_csv(log_path)
    
    print("="*60)
    print(f"{model_type} TRAINING ANALYSIS")
    print("="*60)
    
    # Overall statistics
    print(f"\n📊 Overall Statistics (Episodes 1-{len(df)}):")
    print(f"  Total Episodes: {len(df)}")
    print(f"  Average Reward: {df['total_reward'].mean():.2f} ± {df['total_reward'].std():.2f}")
    print(f"  Max Reward: {df['total_reward'].max():.2f}")
    print(f"  Min Reward: {df['total_reward'].min():.2f}")
    print(f"  Current Epsilon: {df['epsilon'].iloc[-1]:.3f}")
    
    # Recent performance (handle cases with fewer episodes)
    n_recent = min(100, len(df))
    recent_df = df.tail(n_recent)
    print(f"\n📈 Recent Performance (Last {n_recent} Episodes):")
    print(f"  Average Reward: {recent_df['total_reward'].mean():.2f} ± {recent_df['total_reward'].std():.2f}")
    print(f"  Max Reward: {recent_df['total_reward'].max():.2f}")
    
    # Check if learning is happening
    print(f"\n🧠 Learning Indicators:")
    print(f"  Average Loss: {df['avg_loss'].mean():.4f}")
    print(f"  Average Q-value: {df['avg_q_value'].mean():.2f}")
    n_recent_q = min(100, len(df))
    print(f"  Q-value Trend: {df['avg_q_value'].iloc[-n_recent_q:].mean():.2f} (last {n_recent_q} eps)")
    
    # Check for improvement over time (only if enough episodes)
    if len(df) >= 100:
        first_100 = df.head(100)['total_reward'].mean()
        last_100 = df.tail(100)['total_reward'].mean()
        improvement = ((last_100 - first_100) / first_100 * 100) if first_100 != 0 else 0
        
        print(f"\n📊 Progress Check:")
        print(f"  First 100 episodes avg: {first_100:.2f}")
        print(f"  Last 100 episodes avg: {last_100:.2f}")
        print(f"  Improvement: {improvement:+.1f}%")
    else:
        first_half = df.head(len(df)//2)['total_reward'].mean()
        last_half = df.tail(len(df)//2)['total_reward'].mean()
        improvement = ((last_half - first_half) / first_half * 100) if first_half != 0 else 0
        
        print(f"\n📊 Progress Check (Limited Episodes):")
        print(f"  First half avg: {first_half:.2f}")
        print(f"  Last half avg: {last_half:.2f}")
        print(f"  Improvement: {improvement:+.1f}%")
    
    # Diagnosis
    print(f"\n🔍 Diagnosis:")
    if df['epsilon'].iloc[-1] > 0.5:
        print("  ⚠️  Epsilon still high - agent is mostly exploring")
        print("     → This is NORMAL. Rewards will increase when epsilon < 0.3")
    
    if df['avg_q_value'].mean() > 0:
        print("  ✓ Q-values are positive - network is learning")
    else:
        print("  ⚠️  Q-values are low - learning may be slow")
    
    if improvement > 0:
        print(f"  ✓ Rewards improving ({improvement:+.1f}%) - training is working!")
    else:
        print("  ⚠️  No improvement yet - this is normal for early training")
    
    # Expected timeline (adjusted per model type)
    current_eps = len(df)
    print(f"\n⏱️  Expected Timeline for {model_type}:")
    if "VANILLA" in model_type:
        print(f"  Episodes 0-500: Exploration phase (you are here at {current_eps})")
        print(f"  Episodes 500-1500: Gradual improvement expected")
        print(f"  Episodes 1500+: Performance plateau")
        print(f"\n  💡 Vanilla DQN typically needs 1000+ episodes to show clear improvement")
    elif "DOUBLE" in model_type:
        print(f"  Episodes 0-400: Exploration phase (you are here at {current_eps})")
        print(f"  Episodes 400-1200: Gradual improvement expected")
        print(f"  Episodes 1200+: Performance plateau")
        print(f"\n  💡 Double DQN typically learns faster than Vanilla (800+ episodes)")
    elif "DUELING" in model_type:
        print(f"  Episodes 0-400: Exploration phase (you are here at {current_eps})")
        print(f"  Episodes 400-1200: Gradual improvement expected")
        print(f"  Episodes 1200+: Performance plateau")
        print(f"\n  💡 Dueling DQN typically learns faster than Vanilla (800+ episodes)")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if current_eps < 1000:
        print("  → Continue training to at least 1000 episodes")
        print("  → Monitor Q-values (should gradually increase)")
        print("  → Check again when epsilon < 0.3")
    
    print("\n" + "="*60)
    
    return df

if __name__ == "__main__":
    # Analyze all three models
    vanilla_df = analyze_training_progress("logs/vanilla_dqn/training_log.csv")
    double_df = analyze_training_progress("logs/double_dqn/training_log.csv")
    dueling_df = analyze_training_progress("logs/dueling_dqn/training_log.csv")
