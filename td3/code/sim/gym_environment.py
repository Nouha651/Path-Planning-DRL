import numpy as np
import matplotlib.pyplot as plt

class ContinuousEnv:
    def __init__(self):
        self.agent_pos = np.array([0.2, 0.2])  # Start agent at (0, 0)
        self.goal_pos = np.array([4.75, 4.75])   # Fixed goal at (5, 5)
        self.obstacles = [np.array([2.0, 2.0]), np.array([3.5, 3.5])]  # Add two obstacles
        self.agent_radius = 0.2 # Define agent's size (radius)
        self.obstacle_radius = 0.2  # Define obstacle's size (radius)
        self.goal_radius = 0.2  # Define goal's size (radius)

        self.fig, self.ax = plt.subplots()

        # Set boundaries for the environment
        self.boundaries = np.array([[-5, -5], [5, 5]])  # min boundaries (-5, -5) and max boundaries (5, 5)

    def reset(self):
        # Set a fixed starting position for the agent
        self.agent_pos = np.array([0.0, 0.0])

        if np.allclose(self.agent_pos, [0.0, 0.0], atol=1e-2):
            print("Agent starts at position (0, 0) in reset")

        return self.agent_pos

    def step(self, action):
        # Store previous distance to the goal
        previous_distance_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)

        # Update the agent's position based on the action
        self.agent_pos += action


        # Calculate new distance to the goal
        distance_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
        done = distance_to_goal < 0.1  # If within 0.1 units of the goal, episode ends

        # Initialize reward
        reward = 0

        # Goal achievement
        if done:
            reward += 100  # Large positive reward for reaching the goal
            print("goal met and done is:", done)
        else:
            # Reward or penalize based on progress towards the goal
            if distance_to_goal < previous_distance_to_goal:
                reward += .1  # Reward for moving closer to the goal
            else:
                reward -= .1  # Penalty for moving further from the goal

            # Add penalties if agent hits an obstacle
            for obstacle in self.obstacles:
                distance_to_obstacle = np.linalg.norm(self.agent_pos - obstacle)
                if distance_to_obstacle < 0.5:
                    reward -= 50  # Large penalty for hitting an obstacle
                    done = True  # End episode if obstacle is hit
                    print("There is an obstacle and done is:", done)

                #elif 0.2 <= distance_to_obstacle < 0.5:
                #    reward += 2  # Small positive reward for avoiding the obstacle closely
            #print("There is an obstacle and done is:", done, " out of the for loop")
            # Check if the agent hits the wall (boundaries)
            if np.any(self.agent_pos < self.boundaries[0]) or np.any(self.agent_pos > self.boundaries[1]):
                reward -= 20  # Large penalty for hitting the boundary
                done = True  # End episode if the agent hits the wall
                print("There is a wall and done is:", done)

        return self.agent_pos, reward, done

    def render(self):
        self.ax.clear()
        # Set the extended view limits to [-10, 10]
        self.ax.set_xlim([-10, 10])
        self.ax.set_ylim([-10, 10])

        # Draw the goal with specified size
        goal_circle = plt.Circle(self.goal_pos, self.goal_radius, color='green', label='Goal')
        self.ax.add_patch(goal_circle)

        # Draw the agent with specified size
        agent_circle = plt.Circle(self.agent_pos, self.agent_radius, color='blue', label='Agent')
        self.ax.add_patch(agent_circle)

        # Draw the obstacles with specified size
        for obstacle in self.obstacles:
            obstacle_circle = plt.Circle(obstacle, self.obstacle_radius, color='red', label='Obstacle')
            self.ax.add_patch(obstacle_circle)

        # Ensure that only one label per type is added
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys())

        plt.draw()
        plt.pause(0.01)  # Small pause to update the rendering
