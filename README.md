DQN Agent for Q*bert - Atari Game
Project Overview
This project implements a Deep Q-Network (DQN) agent to learn and play the classic Q*bert Atari 2600 game. Q*bert is an excellent benchmark for deep reinforcement learning because it provides a good balance between complexity and manageability, requiring both spatial reasoning and strategic planning.

Why Q*bert?
Ideal Action Space: 6 discrete actions (makes training tractable)

Rich State Space: Visual input from 210×160 pixel screens

Progressive Difficulty: 9+ levels with increasing complexity

Excellent Benchmark Data: Abundant research data from 2016-2024 for validation

Clear Reward Signal: Each cube color change = +25 points

Strategic Learning: Requires planning and spatial awareness, not just reflex

Game Mechanics
Objective
Change the color of all cubes in the pyramid to the target color (shown in top-left) by hopping on them. When all cubes match the target color, advance to the next round.

Game Rules
Component	Description
Playfield	3D isometric pyramid of 28 cubes
Controls	4-way joystick (+ 2 special actions = 6 total actions)
Level 1	Hop once on each cube to change color
Level 2	Hop twice on each cube (intermediate → target color)
Level 3+	Complex color mechanics; later cubes revert if stepped again
Enemies	Coily (snake), Ugg, Wrong Way, Slick, Sam
Mechanics That Make This Challenging for RL
Dynamic Cube Behavior:

Early levels (1-2): Simple one-touch or two-touch mechanics

Later levels (5+): Cubes revert to original color if touched again after completion

Creates non-stationary environment where policy must adapt per level

Enemy Patterns:

Coily actively pursues Q*bert

Requires both evasion and engagement strategies

Different speeds and behaviors per level

Escape Mechanics:

Floating discs teleport Q*bert to pyramid top

If Coily follows, he falls and dies (all enemies disappear)

Strategic use improves scores significantly

Reward Structure
Action	Points
Change cube color	+25
Catch green ball	+100
Defeat enemy (Slick/Sam)	+300
Defeat Coily via disc	+500
Screen completion bonus	1,000 (Level 1) + 250 per level
Unused discs at level end	50-100 per disc
Benchmark Performance Data
Vanilla DQN vs Double DQN vs Dueling DQN
Historical results from Stanford CS 2024 research:

Algorithm	Score	Episodes	Improvement
Vanilla DQN	734	3,601	Baseline
Double DQN	1,428	4,718	+94%
Dueling DQN	2,256	6,369	+58% (over Double)
Key Observations:

Training curves show clear oscillations (agent adapting to each level's new rules)

Double DQN reduces overestimation bias significantly

Dueling DQN further improves by separating value and advantage functions

Scores still increasing at 6,369 episodes (no plateau reached)

Historical Baseline (2016 - Stanford CS229)
Algorithm	Score
Vanilla DQN	700
DRQN (Recurrent)	850
Implementation Requirements
Environment Setup
bash
# Install dependencies
pip install gym[atari]
pip install ale-py
pip install torch torchvision torchaudio
pip install numpy matplotlib

# Optional: For Atari ROM files
pip install autorom[accept-rom-license]
Game Environment
python
import gym
env = gym.make("Qbert-v4")  # or "Qbert-ram-v4" for RAM input

# Action space
# 0: NOOP
# 1: FIRE
# 2: UP
# 3: RIGHT
# 4: LEFT
# 5: DOWN
Expected Training Timeline
Hardware	DQN (Vanilla)	Double DQN	Dueling DQN
GPU (RTX 3080+)	2-3 hours	3-4 hours	4-5 hours
GPU (RTX 2080)	4-6 hours	6-8 hours	8-10 hours
CPU Only	24-48 hours	Not recommended	Not recommended
Project Structure
text
dqn-qbert/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.py                 # Hyperparameters
├── model.py                  # DQN network architecture
├── agent.py                  # DQN agent implementation
├── environment.py            # Environment wrapper
├── train.py                  # Training loop
├── evaluate.py               # Evaluation script
├── utils/
│   ├── replay_buffer.py      # Experience replay buffer
│   ├── preprocessing.py      # Frame preprocessing
│   └── visualization.py      # Plot results
└── checkpoints/              # Saved models
Quick Start
1. Basic Training
python
python train.py --algorithm dqn --episodes 5000
2. Compare Algorithms
python
# Train all three variants
python train.py --algorithm dqn
python train.py --algorithm double_dqn
python train.py --algorithm dueling_dqn
3. Evaluate Trained Agent
python
python evaluate.py --checkpoint checkpoints/dqn_best.pth --episodes 10
Key Hyperparameters to Tune
Parameter	Default	Range	Notes
Learning Rate	0.0001	[1e-5, 1e-3]	Standard: 1e-4
Discount Factor (γ)	0.99	[0.95, 0.999]	Higher = longer horizon
Epsilon Start	1.0	[0.5, 1.0]	Exploration rate
Epsilon Final	0.01	[0.001, 0.1]	Minimum exploration
Epsilon Decay	500k	[100k, 1M]	Decay schedule
Replay Buffer Size	100k	[50k, 1M]	Memory usage
Batch Size	32	[16, 64]	Gradient step size
Update Frequency	4	[1, 10]	Steps between updates
Target Update	1000	[500, 5000]	Sync target network
Expected Challenges & Solutions
Challenge 1: High Variance in Rewards
Problem: Q*bert has oscillating rewards when rules change each level
Solution: Use target networks and double Q-learning to stabilize training

Challenge 2: Long Convergence Time
Problem: Needs 3,600+ episodes for vanilla DQN
Solution: Consider Double DQN or Dueling DQN variants

Challenge 3: Exploration vs Exploitation
Problem: ε-greedy may not explore efficiently
Solution: Use Prioritized Experience Replay (see improvements section)

Challenge 4: Memory Constraints
Problem: Large replay buffer + network memory
Solution: Use frame stacking (4 frames) and grayscale preprocessing

Validation Metrics
Compare your results against benchmarks:

python
# Target scores to validate implementation
targets = {
    "vanilla_dqn": 734,      # ± 100
    "double_dqn": 1428,      # ± 150
    "dueling_dqn": 2256,     # ± 200
}
If your scores are significantly lower, check:

✓ Preprocessing: Frame stacking, grayscale, normalization

✓ Network architecture: Conv2d layers matching original

✓ Hyperparameters: Learning rate, update frequencies

✓ Replay buffer: Experience sampling correctness

✓ Target network: Synchronization schedule

Literature & References
Original Papers
DQN (2015): Mnih et al. "Human-level control through deep reinforcement learning" - Nature

Double DQN (2015): Van Hasselt et al. "Deep Reinforcement Learning with Double Q-learning"

Dueling DQN (2015): Wang et al. "Dueling Network Architectures for Deep Reinforcement Learning"

Benchmark Studies
Stanford CS229 (2016): Deep Q-Learning with Recurrent Neural Networks

Stanford CS224R (2024): Q*bert Baseline Performance Comparison

Empirical Study (2024): Comparative Study of Atari Algorithms

Implementation Guides
PyTorch DQN Tutorial: https://pytorch.org/tutorials/

OpenAI Spinning Up: https://spinningup.openai.com/

DeepMind Blog: https://deepmind.google/blog/

Improvements & Extensions
Level 1: Core Implementation ✓
 Vanilla DQN

 Experience Replay

 Target Network

Level 2: Algorithm Improvements
 Double DQN (reduce overestimation bias)

 Dueling DQN (separate value/advantage)

 Prioritized Experience Replay (sample important transitions)

Level 3: Advanced Methods
 Rainbow DQN (combine all improvements)

 DRQN (add LSTM for partial observability)

 Noisy Networks (exploration via parameter noise)

Level 4: Analysis & Visualization
 Learning curves with confidence intervals

 Attention heatmaps on game frames

 Action value heatmaps

 Video recordings of agent gameplay

Performance Notes
Why Q*bert is Harder Than Pong
Larger Action Space: 6 actions vs 3 (Pong)

Complex State Transitions: Levels change rules mid-training

Multiple Enemies: Requires multi-objective reasoning

Longer Time Horizon: Needs credit assignment over 50+ steps

Partial Observability: Can't see all enemies at once

Training Stability Tips
Use Double DQN: Reduces Q-value overestimation which causes instability

Monitor Variance: Watch for sudden score drops (sign of divergence)

Log Frequently: Save checkpoints every 100 episodes

Validate Early: Run evaluation every 500 episodes

Decay Exploration: ε should decrease smoothly, not suddenly

Troubleshooting
Agent learns to "camp" in corners
Cause: Reward shaping issue; disc mechanics not implemented

Fix: Verify disc logic; check enemy collision detection

Training diverges (NaNs in loss)
Cause: Learning rate too high or reward scaling issues

Fix: Reduce LR to 1e-5; normalize rewards to [-1, 1]

Agent stuck at low score after 1000 episodes
Cause: Insufficient exploration or local maxima

Fix: Increase initial ε or add noise to action selection

Memory errors on smaller GPUs
Cause: Replay buffer too large or batch size too big

Fix: Reduce buffer from 100k to 50k; reduce batch size to 16

Citation
If you use this code in research, please cite:

text
@article{mnih2015human,
  title={Human-level control through deep reinforcement learning},
  author={Mnih, Volodymyr and others},
  journal={Nature},
  year={2015}
}

@misc{dqn_qbert_2026,
  title={DQN Agent for Q*bert},
  author={Your Name},
  year={2026},
  howpublished={\url{https://github.com/yourusername/dqn-qbert}}
}
License
MIT License - See LICENSE file for details

Contact & Support
For questions or issues:

Check the Troubleshooting section

Review the benchmark comparisons

Open an issue on GitHub with:

Your hyperparameters

Training logs (first 10 lines)

Hardware specs

Expected vs actual performance
