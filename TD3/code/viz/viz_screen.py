import matplotlib.pyplot as plt

class Visualize:
    def __init__(self):
        self.fig, self.ax = plt.subplots()

    def render(self, agent_pos, target_pos):
        self.ax.clear()
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])

        # Plot agent and target
        self.ax.scatter(agent_pos[0], agent_pos[1], c='blue', label='Agent')
        self.ax.scatter(target_pos[0], target_pos[1], c='red', label='Target')

        plt.pause(0.01)  # Short pause to create animation
