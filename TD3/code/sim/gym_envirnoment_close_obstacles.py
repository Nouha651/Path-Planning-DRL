import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # Import for displaying images

class ContinuousEnv:
    def __init__(self):
        self.agent_pos = np.array([0.2, 0.2])  # Start agent at (0.2, 0.2)
        self.start_pos = np.array([0.0, 0.0])  # Define the start position at (0, 0)
        self.goal_pos = np.array([4.3, 5])   # Fixed goal at (5, 5)
        self.obstacles = [np.array([-4.0, -4.0]), np.array([3.5, 3]), np.array([7, 7]), np.array([-5.4, 5.4])]  # Add obstacles
        self.agent_radius = 0.2  # Define agent's size (radius)
        self.obstacle_radius = 0.2  # Define obstacle's size (radius)
        self.goal_radius = 0.4  # Define goal's size (radius)

        self.fig, self.ax = plt.subplots()

        # Set boundaries for the environment
        self.boundaries = np.array([[-6, -6], [8, 8]])  # min boundaries (-5, -5) and max boundaries (5, 5)

    def reset(self):
        self.agent_pos = np.array([0.0, 0.0])
        return self.agent_pos

    def step(self, action):
        previous_pos = self.agent_pos.copy()  # Store the previous position of the agent
        self.agent_pos += action  # Update the agent's position based on the action

        distance_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
        done = distance_to_goal < 0.4  # Check if the agent has reached the goal

        reward = 0
        if done:
            reward += 5000
            print("GOALLL", done)
        else:
            previous_distance_to_goal = np.linalg.norm(previous_pos - self.goal_pos)
            # Progress reward or penalty based on distance to goal
            if distance_to_goal < previous_distance_to_goal:
                reward -= 0.1 + 0.1 * (previous_distance_to_goal - distance_to_goal)
            else:
                reward -= 1 + (distance_to_goal - previous_distance_to_goal)

            # Check if the agent passed through any obstacles
            for obstacle in self.obstacles:
                dist_to_obstacle = np.linalg.norm(
                    np.cross(self.agent_pos - previous_pos, previous_pos - obstacle)) / np.linalg.norm(
                    self.agent_pos - previous_pos)
                if dist_to_obstacle < self.obstacle_radius:
                    reward -= 1000  # Large penalty for passing through an obstacle
                    done = True
                    break

            # Boundary check
            if np.any(self.agent_pos < self.boundaries[0]) or np.any(self.agent_pos > self.boundaries[1]):
                reward -= 1000  # Penalty for hitting the boundary
                done = True

        return self.agent_pos, reward, done

    def render(self, path=None, optimal_path=None,
               goal_image_path="/home/nouha/pfe/TD3+DQN+SAC/TD3/code/icons/goal_icon.png"):
        def on_key(event):
            if event.key == 'q':
                plt.close()

        plt.gcf().canvas.mpl_connect('key_press_event', on_key)

        self.ax.clear()

        # Draw environment boundaries
        # Comment out these lines to hide the walls in the rendering
        # boundary_lines = [
        #     [self.boundaries[0][0], self.boundaries[0][1]],  # Bottom-left corner
        #     [self.boundaries[1][0], self.boundaries[0][1]],  # Bottom-right corner
        #     [self.boundaries[1][0], self.boundaries[1][1]],  # Top-right corner
        #     [self.boundaries[0][0], self.boundaries[1][1]],  # Top-left corner
        #     [self.boundaries[0][0], self.boundaries[0][1]]   # Close the rectangle
        # ]
        # boundary_lines = np.array(boundary_lines)
        # self.ax.plot(boundary_lines[:, 0], boundary_lines[:, 1], 'k-', linewidth=2, label='Walls')

        self.ax.set_xlim([-10, 10])
        self.ax.set_ylim([-10, 10])

        # Add start point as a cyan circle
        start_circle = plt.Circle((float(self.start_pos[0]), float(self.start_pos[1])), 0.3, color='cyan',
                                  label='Start')
        self.ax.add_patch(start_circle)

        # Add goal as an image if path provided, otherwise as a green circle
        if goal_image_path is not None:
            goal_image = mpimg.imread(goal_image_path)
            self.ax.imshow(goal_image, extent=(
                self.goal_pos[0] - self.goal_radius, self.goal_pos[0] + self.goal_radius,
                self.goal_pos[1] - self.goal_radius, self.goal_pos[1] + self.goal_radius
            ))
        else:
            goal_circle = plt.Circle((float(self.goal_pos[0]), float(self.goal_pos[1])), self.goal_radius,
                                     color='green', label='Goal')
            self.ax.add_patch(goal_circle)

        # Add agent as a blue circle
        agent_circle = plt.Circle((float(self.agent_pos[0]), float(self.agent_pos[1])), self.agent_radius, color='blue',
                                  label='Agent')
        self.ax.add_patch(agent_circle)

        # Add obstacles as red circles
        for obstacle in self.obstacles:
            obstacle_circle = plt.Circle((float(obstacle[0]), float(obstacle[1])), self.obstacle_radius, color='red',
                                         label='Obstacle')
            self.ax.add_patch(obstacle_circle)

        # Draw the TD3 path if provided
        if path is not None:
            path = np.array(path)
            self.ax.plot(path[:, 0], path[:, 1], linestyle='--', color='blue', label='TD3 Path')

        # Draw the RRT* smart optimal path if provided
        if optimal_path is not None:
            optimal_path = np.array(optimal_path)
            self.ax.plot(optimal_path[:, 0], optimal_path[:, 1], linestyle='--', color='purple',
                         label='RRT*_Smart Path')

        # Handle legend to avoid duplicate labels
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys())

        plt.draw()
        plt.pause(0.5)
