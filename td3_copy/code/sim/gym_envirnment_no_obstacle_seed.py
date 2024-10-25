import numpy as np
import matplotlib.pyplot as plt


class ContinuousEnv:
    def __init__(self):
        self.agent_pos = np.array([0, 0])  # Start agent at (0, 0)
        self.goal_pos = np.array([5, 5])  # Fixed goal at (5, 5)
        self.agent_radius = 0.2  # Define agent's size (radius)
        self.obstacle_radius = 0.2  # Define obstacle's size (radius)
        self.goal_radius = 0.4  # Define goal's size (radius)

        self.fig, self.ax = plt.subplots()

        # Set boundaries for the environment
        self.boundaries = np.array([[-6, -6], [8, 8]])  # min boundaries (-5, -5) and max boundaries (5, 5)

    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self):
        # Set a fixed starting position for the agent
        self.agent_pos = np.array([0.0, 0.0])
        return self.agent_pos

    def step(self, action):
        previous_distance_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
        self.agent_pos += action
        distance_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
        done = distance_to_goal < 0.4

        reward = 5000 if done else -1 * (distance_to_goal >= previous_distance_to_goal)
        if np.any(self.agent_pos < self.boundaries[0]) or np.any(self.agent_pos > self.boundaries[1]):
            reward -= 50
            done = True

        return self.agent_pos, reward, done

    def render(self, path=None):
        def on_key(event):
            if event.key == 'q':
                plt.close()

        plt.gcf().canvas.mpl_connect('key_press_event', on_key)
        self.ax.clear()
        self.ax.set_xlim([-10, 10])
        self.ax.set_ylim([-10, 10])

        goal_circle = plt.Circle((float(self.goal_pos[0]), float(self.goal_pos[1])), self.goal_radius, color='green',
                                 label='Goal')
        agent_circle = plt.Circle((float(self.agent_pos[0]), float(self.agent_pos[1])), self.agent_radius, color='blue',
                                  label='Agent')
        self.ax.add_patch(goal_circle)
        self.ax.add_patch(agent_circle)

        if path is not None:
            path = np.array(path)
            self.ax.plot(path[:, 0], path[:, 1], linestyle='--', color='blue', label='Path')

        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys())

        plt.draw()
        plt.pause(0.1)
