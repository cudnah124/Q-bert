"""
Quick analysis script to check if DQN training is progressing normally
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_training_progress(log_path="logs/vanilla_dqn/training_log.csv"):
    """Analyze training metrics to check learning progress"""
    
def plot_training_metrics(df, model_type, log_dir):
    """Plot training metrics: Reward, Loss, Q-values, Epsilon"""
    # Create plots directory if it doesn't exist
    plot_dir = os.path.join(log_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"{model_type} Training Analysis", fontsize=16)
    
    # 1. Total Reward vs Episode (with Moving Average)
    axes[0, 0].plot(df['episode'], df['total_reward'], alpha=0.3, color='blue', label='Per Episode')
    if len(df) >= 50:
        window = min(100, len(df)//2)
        sma = df['total_reward'].rolling(window=window).mean()
        axes[0, 0].plot(df['episode'], sma, color='red', linewidth=2, label=f'SMA {window}')
    axes[0, 0].set_title("Total Reward per Episode")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Average Loss vs Episode
    axes[0, 1].plot(df['episode'], df['avg_loss'], color='orange')
    axes[0, 1].set_title("Average Loss per Episode")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_yscale('log') # Loss is often better viewed on log scale
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Average Q-Value vs Episode
    axes[1, 0].plot(df['episode'], df['avg_q_value'], color='green')
    axes[1, 0].set_title("Average Q-Value per Episode")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Q-Value")
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Epsilon vs Episode
    axes[1, 1].plot(df['episode'], df['epsilon'], color='purple')
    axes[1, 1].set_title("Exploration Rate (Epsilon)")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Epsilon")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    save_path = os.path.join(plot_dir, "training_metrics.png")
    plt.savefig(save_path)
    print(f"📈 Plot saved to: {save_path}")
    
    # Also save a separate Reward Plot for cleaner view if requested
    plt.figure(figsize=(10, 6))
    plt.plot(df['episode'], df['total_reward'], alpha=0.3, color='blue', label='Per Episode')
    if len(df) >= 50:
        window = min(100, len(df)//2)
        sma = df['total_reward'].rolling(window=window).mean()
        plt.plot(df['episode'], sma, color='red', linewidth=2, label=f'SMA {window}')
    plt.title(f"{model_type}: Total Reward Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    reward_plot_path = os.path.join(plot_dir, "reward_chart.png")
    plt.savefig(reward_plot_path)
    plt.close('all') # Close all figures to free memory
    
    return save_path

def analyze_training_progress(log_path="logs/vanilla_dqn/training_log.csv"):
    """Analyze training metrics to check learning progress"""
    
    if not os.path.exists(log_path):
        print(f"❌ Log file not found: {log_path}")
        return None
    
    log_dir = os.path.dirname(log_path)
    
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
        first_half = df.head(len(df)//2)['total_reward'].mean() if len(df) > 1 else 0
        last_half = df.tail(len(df)//2)['total_reward'].mean() if len(df) > 1 else 0
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
    
    # Plotting
    plot_training_metrics(df, model_type, log_dir)
    
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
    prioritized_df = analyze_training_progress("logs/prioritized_dqn/training_log.csv")

