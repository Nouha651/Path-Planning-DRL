import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.animation as animation

class PathPlanningEnv(gym.Env):
    def __init__(self, grid_size=5, start=(0, 0), goal=(4, 4), obstacles=None, step_penalty=-0.1, obstacle_penalty=-2,
                 goal_reward=1000, random_obstacles=False, seed=None):
        super(PathPlanningEnv, self).__init__()
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.state = self.start
        self.random_obstacles = random_obstacles
        self.seed = seed
        if random_obstacles:
            np.random.seed(self.seed)
            self.obstacles = self._generate_random_obstacles()
        else:
            self.obstacles = obstacles if obstacles else []
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=grid_size - 1, shape=(2,), dtype=np.int32)
        self.steps_taken = 0
        self.step_penalty = step_penalty
        self.obstacle_penalty = obstacle_penalty
        self.goal_reward = goal_reward

    def _generate_random_obstacles(self):
        num_obstacles = np.random.randint(10, 15)
        obstacles = []
        while len(obstacles) < num_obstacles:
            x, y = np.random.randint(0, self.grid_size, size=2)
            if (x, y) not in obstacles and (x, y) != self.start and (x, y) != self.goal:
                obstacles.append((x, y))
        return obstacles

    def reset(self):
        self.state = self.start
        self.steps_taken = 0
        if self.random_obstacles:
            np.random.seed(self.seed)
            self.obstacles = self._generate_random_obstacles()
        return np.array(self.state)

    def step(self, action):
        x, y = self.state
        if action == 0:  # up
            y = max(0, y - 1)
        elif action == 1:  # down
            y = min(self.grid_size - 1, y + 1)
        elif action == 2:  # left
            x = max(0, x - 1)
        elif action == 3:  # right
            x = min(self.grid_size - 1, x + 1)

        next_state = (x, y)
        self.steps_taken += 1

        if next_state in self.obstacles:
            reward = self.obstacle_penalty
            next_state = self.state
            done = False
        elif next_state == self.goal:
            reward = self.goal_reward
            done = True
        else:
            reward = self.step_penalty
            done = False

        self.state = next_state
        return np.array(self.state), reward, done, {}

    def render(self, path=None):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)

        # Draw grid
        for x in range(self.grid_size + 1):
            ax.axhline(x, lw=2, color='black', zorder=0)
            ax.axvline(x, lw=2, color='black', zorder=0)

        # Draw start
        ax.add_patch(patches.Rectangle(self.start, 1, 1, edgecolor='blue', facecolor='lightblue', lw=2, label='Start'))

        # Draw goal
        goal_icon = self._load_icon("icons/goal_icon.png", zoom=0.6)
        goal_artist = AnnotationBbox(goal_icon, (self.goal[0] + 0.5, self.goal[1] + 0.5), frameon=False)
        ax.add_artist(goal_artist)

        # Draw obstacles
        for obs in self.obstacles:
            ax.add_patch(patches.Rectangle(obs, 1, 1, edgecolor='red', facecolor='darkred', lw=2, label='Obstacle'))

        # Load agent icon
        agent_icon = self._load_icon("icons/agent_icon.png", zoom=0.6)

        # If path is provided, animate the agent moving along the path
        if path:
            agent_position = path[0]
            agent_artist = AnnotationBbox(agent_icon, (agent_position[0] + 0.5, agent_position[1] + 0.5), frameon=False)
            ax.add_artist(agent_artist)

            def update(num, path, agent_artist, ax):
                agent_position = path[num]
                agent_artist.xybox = (agent_position[0] + 0.5, agent_position[1] + 0.5)
                ax.plot(agent_position[0] + 0.5, agent_position[1] + 0.5, 'bo', markersize=8)  # Larger blue dots
                return agent_artist,

            ani = animation.FuncAnimation(fig, update, frames=len(path), fargs=[path, agent_artist, ax],
                                          interval=500, repeat=False)

        plt.gca().invert_yaxis()  # Invert Y-axis to match the grid indexing with matrix indexing
        plt.show()

    def _load_icon(self, path, zoom=0.5):
        return OffsetImage(plt.imread(path), zoom=zoom)
