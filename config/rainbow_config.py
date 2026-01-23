# Rainbow DQN Configuration

# Environment
FRAME_STACK = 4
FRAME_WIDTH = 84
FRAME_HEIGHT = 84
NUM_ACTIONS = 6

# Learning
LEARNING_RATE = 0.0001
GAMMA = 0.99
BATCH_SIZE = 32

# Update frequencies
UPDATE_FREQUENCY = 4
TARGET_UPDATE = 5000

# NO epsilon (Noisy Networks handle exploration)
# Rainbow uses parameter space noise instead of epsilon-greedy

# Prioritized Replay Buffer
REPLAY_BUFFER_SIZE = 100000
ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 100000
PRIORITY_EPSILON = 0.01

# Multi-step Learning
N_STEPS = 3  # Use 3-step returns

# Distributional RL (C51)
NUM_ATOMS = 51  # Number of atoms in value distribution
V_MIN = -10  # Minimum value
V_MAX = 10   # Maximum value

# Noisy Networks
SIGMA_INIT = 0.5  # Initial noise parameter

# Training
TRAIN_START = 10000
MAX_EPISODES = 6000  # Rainbow typically needs more episodes
EVAL_FREQUENCY = 100
SAVE_FREQUENCY = 500

# Device
DEVICE = "cuda"
