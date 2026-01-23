from models.vanilla_dqn import VanillaDQN


class PrioritizedDQN(VanillaDQN):
    """
    Prioritized DQN uses the same network architecture as Vanilla DQN.
    
    The improvement comes from the prioritized experience replay buffer
    which samples important transitions more frequently.
    
    This class is an alias for VanillaDQN - the actual prioritization
    logic is handled by PrioritizedReplayBuffer and PrioritizedDQNAgent.
    """
    pass
