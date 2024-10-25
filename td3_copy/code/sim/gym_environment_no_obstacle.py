import numpy as np
import matplotlib.pyplot as plt

class ContinuousEnv:
    def __init__(self):
        self.agent_pos = np.array([0, 0])  # Start agent at (0, 0)
        self.goal_pos = np.array([5, 5])   # Fixed goal at (5, 5)
        self.agent_radius = 0.2 # Define agent's size (radius)
        self.obstacle_radius = 0.2  # Define obstacle's size (radius)
        self.goal_radius = 0.4  # Define goal's size (radius)

        self.fig, self.ax = plt.subplots()

        # Set boundaries for the environment
        self.boundaries = np.array([[-6, -6], [8, 8]])  # min boundaries (-5, -5) and max boundaries (5, 5)

    def reset(self):
        # Set a fixed starting position for the agent
        self.agent_pos = np.array([0.0, 0.0])

        #if np.allclose(self.agent_pos, [0.0, 0.0], atol=1e-2):
            #print("Agent starts at position (0, 0) in reset")

        return self.agent_pos

    def step(self, action):
        # Store previous distance to the goal
        previous_distance_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)

        # Update the agent's position based on the action
        self.agent_pos += action


        # Calculate new distance to the goal
        distance_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
        done = distance_to_goal < 0.4  # If within 0.1 units of the goal, episode ends

        # Initialize reward
        reward = 0

        # Goal achievement
        if done:
            reward += 5000  # Large positive reward for reaching the goal
            print("GOALLL ", done)
        else:
            # Reward or penalize based on progress towards the goal
            if distance_to_goal < previous_distance_to_goal:
                reward -= 0.1 + 0.1 * (
                            previous_distance_to_goal - distance_to_goal)  # Small negative reward for getting closer
            else:
                reward -= 1 + (distance_to_goal - previous_distance_to_goal)  # Larger penalty for moving further


            # Check if the agent hits the wall (boundaries)
            if np.any(self.agent_pos < self.boundaries[0]) or np.any(self.agent_pos > self.boundaries[1]):
                reward -= 50  # Large penalty for hitting the boundary
                done = True  # End episode if the agent hits the wall
                #print("WALL")

        return self.agent_pos, reward, done

    import matplotlib.pyplot as plt

    def render(self, path=None):
        # Close the window when 'q' is pressed
        def on_key(event):
            if event.key == 'q':
                plt.close()

        # Set the event connection to listen for key press events
        plt.gcf().canvas.mpl_connect('key_press_event', on_key)

        self.ax.clear()

        # Set the extended view limits to [-10, 10]
        self.ax.set_xlim([-10, 10])
        self.ax.set_ylim([-10, 10])

        # Draw the goal with the specified size (convert position array to tuple of floats)
        goal_circle = plt.Circle((float(self.goal_pos[0]), float(self.goal_pos[1])), self.goal_radius, color='green',
                                 label='Goal')
        self.ax.add_patch(goal_circle)

        # Draw the agent with the specified size (convert position array to tuple of floats)
        agent_circle = plt.Circle((float(self.agent_pos[0]), float(self.agent_pos[1])), self.agent_radius, color='blue',
                                  label='Agent')
        self.ax.add_patch(agent_circle)


        # If a path is provided, draw the path as a dashed line
        if path is not None:
            path = np.array(path)
            self.ax.plot(path[:, 0], path[:, 1], linestyle='--', color='blue', label='Path')

        # Ensure that only one label per type is added
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys())

        plt.draw()
        plt.pause(0.1)  # Pause for a short duration to control the simulation speed
