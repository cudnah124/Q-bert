# DQN Agent for Q*bert - Atari Game

Deep Q-Network (DQN) implementation for learning to play the classic Q*bert Atari 2600 game using reinforcement learning.

## Table of Contents
- [About Q*bert](#about-qbert)
- [Why This Game? ](#why-this-game)
- [Game Mechanics](#game-mechanics)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Benchmarks](#benchmarks)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## About Q*bert

Q*bert is an excellent benchmark for deep reinforcement learning with: 
- **6 discrete actions** - tractable action space
- **Visual input** - 210×160 pixel screens
- **Progressive difficulty** - 9+ levels with increasing complexity
- **Clear rewards** - +25 points per cube color change
- **Strategic gameplay** - requires planning and spatial awareness

## Why This Game?

Q*bert provides unique challenges for RL agents: 

1. **Dynamic Environment**: Game rules change between levels
2. **Multi-objective**:  Must balance cube completion with enemy evasion
3. **Long-term Planning**: Strategic disc usage for +500 point bonuses
4. **Non-stationary**: Later levels revert cube colors if stepped on again

## Game Mechanics

### Objective
Change all pyramid cubes to the target color (shown top-left) by hopping on them.

### Reward Structure
| Action | Points |
|--------|--------|
| Change cube color | +25 |
| Catch green ball | +100 |
| Defeat Slick/Sam | +300 |
| Defeat Coily via disc | +500 |
| Complete screen | 1,000 + (250 × level) |
| Unused disc | 50-100 |

### Level Progression
- **Level 1**: Hop once to change color
- **Level 2**: Hop twice (intermediate → target)
- **Level 3+**: Cubes revert if stepped on again after completion

### Enemies
- **Coily** (snake): Actively pursues Q*bert
- **Ugg & Wrong Way**:  Move horizontally
- **Slick & Sam**:  Change cube colors back

## Installation

```bash
# Install dependencies
pip install gym[atari]
pip install ale-py
pip install torch torchvision torchaudio
pip install numpy matplotlib

# Download Atari ROMs
pip install autorom[accept-rom-license]
```

## 🏃 Quick Start

### 1. Train a Model

```python
# Vanilla DQN
python train_vanilla_dqn.py

# Double DQN
python train_double_dqn.py

# Dueling DQN
python train_dueling_dqn.py

# Prioritized DQN (improved sample efficiency)
python train_prioritized_dqn.py

# Rainbow DQN (state-of-the-art)
python train_rainbow_dqn.py
```

### 2. Evaluate Trained Agent

```python
python evaluate.py --checkpoint checkpoints/dqn_best.pth --episodes 10
```

### 3. Environment Setup

```python
import gym

env = gym.make("Qbert-v4")

# Action space (6 actions)
# 0: NOOP
# 1: FIRE
# 2: UP
# 3: RIGHT
# 4: LEFT
# 5: DOWN
```

## Project Structure

```
Q-bert/
├── README.md                 # This file
├── requirements.txt          # Dependencies
├── config.py                 # Hyperparameters
├── model.py                  # DQN architectures
├── agent.py                  # DQN agent
├── environment.py            # Environment wrapper
├── train.py                  # Training loop
├── evaluate.py               # Evaluation script
├── utils/
│   ├── replay_buffer.py      # Experience replay
│   ├── preprocessing.py      # Frame preprocessing
│   └── visualization.py      # Plotting tools
└── checkpoints/              # Saved models
```

## Benchmarks

### Algorithm Comparison (Stanford CS224R 2024)

| Algorithm | Avg Score | Episodes | Improvement |
|-----------|-----------|----------|-------------|
| Vanilla DQN | 734 | 3,601 | Baseline |
| Double DQN | 1,428 | 4,718 | +94% |
| Dueling DQN | 2,256 | 6,369 | +58% |
| Prioritized DQN | ~1,800-2,000* | 5,000 | +26-40%* |
| Rainbow DQN | ~3,000-4,000* | 6,000 | +67-111%* |

*Estimated based on typical improvements over Dueling DQN

### Historical Baselines (Stanford CS229 2016)

| Algorithm | Avg Score |
|-----------|-----------|
| Vanilla DQN | 700 |
| DRQN (Recurrent) | 850 |

### Expected Training Time

| Hardware | Vanilla DQN | Double DQN | Dueling DQN | Prioritized DQN | Rainbow DQN |
|----------|-------------|------------|-------------|-----------------|-------------|
| RTX 3080+ | 2-3 hours | 3-4 hours | 4-5 hours | 4-6 hours | 8-12 hours |
| RTX 2080 | 4-6 hours | 6-8 hours | 8-10 hours | 8-12 hours | 16-24 hours |
| CPU Only | 24-48 hours | Not recommended | Not recommended | Not recommended | Not recommended |

## Configuration

### Key Hyperparameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| Learning Rate | 0.0001 | [1e-5, 1e-3] | Standard:  1e-4 |
| Discount (γ) | 0.99 | [0.95, 0.999] | Higher = longer horizon |
| Epsilon Start | 1.0 | [0.5, 1.0] | Initial exploration |
| Epsilon Final | 0.01 | [0.001, 0.1] | Min exploration |
| Epsilon Decay | 500k | [100k, 1M] | Decay steps |
| Replay Buffer | 100k | [50k, 1M] | Memory usage |
| Batch Size | 32 | [16, 64] | Gradient samples |
| Update Freq | 4 | [1, 10] | Steps between updates |
| Target Update | 1000 | [500, 5000] | Network sync |

### Validation Targets

```python
targets = {
    "vanilla_dqn": 734,          # ± 100
    "double_dqn": 1428,          # ± 150
    "dueling_dqn": 2256,         # ± 200
    "prioritized_dqn": 1900,     # ± 200 (estimated)
    "rainbow_dqn": 3500,         # ± 500 (estimated)
}
```

If scores are significantly lower, check:
- ✓ Frame preprocessing (stacking, grayscale, normalization)
- ✓ Network architecture (Conv2D layers)
- ✓ Hyperparameters (learning rate, update frequency)
- ✓ Replay buffer implementation
- ✓ Target network synchronization

## Troubleshooting

### Agent camps in corners
- **Cause**: Disc mechanics not implemented properly
- **Fix**: Verify disc logic and enemy collision detection

### Training diverges (NaN loss)
- **Cause**: Learning rate too high or reward scaling issues
- **Fix**: Reduce LR to 1e-5; normalize rewards to [-1, 1]

### Stuck at low score (1000+ episodes)
- **Cause**: Insufficient exploration
- **Fix**: Increase initial ε or add noise to action selection

### Memory errors
- **Cause**: Replay buffer or batch too large
- **Fix**: Reduce buffer to 50k; reduce batch size to 16

## Roadmap

### Level 1: Core Implementation
- [x] Vanilla DQN
- [x] Experience Replay
- [x] Target Network

### Level 2: Algorithm Improvements
- [x] Double DQN
- [x] Dueling DQN
- [x] Prioritized Experience Replay

### Level 3: Advanced Methods
- [x] Rainbow DQN
- [ ] DRQN (LSTM)
- [ ] Distributional RL Visualization

### Level 4: Analysis
- [ ] Learning curves with confidence intervals
- [ ] Attention heatmaps
- [ ] Action value visualization
- [ ] Gameplay recordings

## References

### Original Papers
- **DQN**: Mnih et al. (2015) - [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- **Double DQN**: Van Hasselt et al. (2015) - [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
- **Dueling DQN**: Wang et al. (2015) - [Dueling Network Architectures](https://arxiv.org/abs/1511.06581)
- **Prioritized Replay**: Schaul et al. (2015) - [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- **Rainbow**: Hessel et al. (2017) - [Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298)
- **Noisy Networks**: Fortunato et al. (2017) - [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)
- **C51**: Bellemare et al. (2017) - [A Distributional Perspective on RL](https://arxiv.org/abs/1707.06887)

### Benchmarks
- Stanford CS229 (2016) - Deep Q-Learning with Recurrent Neural Networks
- Stanford CS224R (2024) - Q*bert Baseline Performance Comparison

### Resources
- [PyTorch DQN Tutorial](https://pytorch.org/tutorials/)
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [DeepMind Blog](https://deepmind.google/blog/)

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Contact

For questions or issues: 
1. Check the [Troubleshooting](#troubleshooting) section
2. Review the [Benchmarks](#benchmarks)
3. Open an issue with: 
   - Your hyperparameters
   - Training logs (first 10 lines)
   - Hardware specs
   - Expected vs actual performance

## Citation

```bibtex
@article{mnih2015human,
  title={Human-level control through deep reinforcement learning},
  author={Mnih, Volodymyr and others},
  journal={Nature},
  year={2015}
}

@misc{dqn_qbert_2026,
  title={DQN Agent for Q*bert},
  author={cudnah124},
  year={2026},
  howpublished={\url{https://github.com/cudnah124/Q-bert}}
}
```

---

**Made by PDN for Deep Reinforcement Learning**
