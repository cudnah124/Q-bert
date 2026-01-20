"""
Utility functions for Q*bert environment
"""

def get_level_from_ram(ale):
    """
    Extract the current level/round from Q*bert RAM.
    
    According to Atari documentation:
    - RAM address 56 contains the round/level information
    - The value needs to be interpreted (often starts at 1)
    
    Args:
        ale: The ALE interface from env.unwrapped.ale
        
    Returns:
        int: Current level (1-indexed)
    """
    ram = ale.getRAM()
    # Q*bert level is stored at RAM address 56
    # The game starts at level 1, and the RAM value represents the current round
    level = ram[56] + 1  # Add 1 because RAM is 0-indexed but we want 1-indexed levels
    return max(1, level)  # Ensure minimum level is 1


def get_qbert_info(env):
    """
    Get extended info from Q*bert environment including level.
    
    Args:
        env: Gymnasium environment
        
    Returns:
        dict: Extended info including 'level'
    """
    ale = env.unwrapped.ale
    level = get_level_from_ram(ale)
    return {'level': level}
