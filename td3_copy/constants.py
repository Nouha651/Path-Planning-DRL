# Constants for the environment and agent
GRID_SIZE = 10            # Grid size (not applicable, but for bounding the environment)
MAX_EPISODES = 1000       # Max number of episodes for training
MAX_TIMESTEPS = 1000      # Max timesteps per episode
MAX_ACTION = 1.0          # Max action magnitude for continuous movement
MIN_ACTION = -1.0         # Min action magnitude
STATE_DIM = 2             # State dimension (x, y position)
ACTION_DIM = 2            # Action dimension (delta x, delta y velocity)
