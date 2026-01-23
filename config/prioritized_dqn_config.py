# Prioritized DQN Configuration

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

# Exploration (epsilon-greedy still used)
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_STEPS = 100000

# Prioritized Replay Buffer
REPLAY_BUFFER_SIZE = 100000
ALPHA = 0.6  # Priority exponent (0=uniform, 1=fully prioritized)
BETA_START = 0.4  # Initial importance sampling weight
BETA_FRAMES = 100000  # Frames to anneal beta to 1.0
PRIORITY_EPSILON = 0.01  # Small constant to prevent zero priorities

# Training
TRAIN_START = 10000  # Start training after this many steps
MAX_EPISODES = 5000
EVAL_FREQUENCY = 100
SAVE_FREQUENCY = 500

# Device
DEVICE = "cuda"
