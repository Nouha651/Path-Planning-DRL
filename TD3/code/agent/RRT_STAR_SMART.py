import numpy as np
import math
import random
import matplotlib.pyplot as plt

# Set a random seed for reproducibility
random.seed(26)
np.random.seed(26)


class Node:
    def __init__(self, coord, start_node=False, target_node=False):
        self.x = coord[0]
        self.y = coord[1]
        self.parent = None
        self.children = []
        self.cost = 1e7
        self.start_node = start_node
        self.target_node = target_node
        if self.start_node:
            self.cost = 0

    def get_position(self):
        return np.array([self.x, self.y])


class RRTStarSmart:
    def __init__(self, env, max_iterations=1000, search_radius=2.0, epsilon=1.0, rewiring_radius=3.0):
        self.env = env
        self.start = Node(self.env.start_pos, start_node=True)
        self.goal = Node(self.env.goal_pos, target_node=True)
        self.nodes = [self.start]
        self.max_iterations = max_iterations
        self.search_radius = search_radius
        self.epsilon = epsilon
        self.rewiring_radius = rewiring_radius
        self.path = []

    def is_collision_free(self, point):
        for obstacle in self.env.obstacles:
            distance = np.linalg.norm(point - obstacle)
            if distance < self.env.obstacle_radius:
                return False
        return True

    def find_nearest_node(self, point):
        distances = [np.linalg.norm(node.get_position() - point) for node in self.nodes]
        nearest_index = np.argmin(distances)
        return self.nodes[nearest_index]

    def steer(self, from_node, to_point):
        direction = to_point - from_node.get_position()
        distance = np.linalg.norm(direction)
        if distance > self.epsilon:
            direction = direction / distance * self.epsilon
        new_point = from_node.get_position() + direction
        return new_point if self.is_collision_free(new_point) else None

    def find_proximal_node(self, new_node):
        proximal_node = None
        for node in self.nodes:
            dist = np.linalg.norm(new_node.get_position() - node.get_position())
            if dist < self.rewiring_radius:
                if node.cost + dist < new_node.cost:
                    proximal_node = node
                    new_node.cost = node.cost + dist
                    new_node.parent = proximal_node
        if proximal_node:
            proximal_node.children.append(new_node)
        return new_node, proximal_node is not None

    def rewire_nodes(self, new_node):
        for node in self.nodes:
            dist = np.linalg.norm(new_node.get_position() - node.get_position())
            if dist < self.rewiring_radius and new_node.cost + dist < node.cost:
                if node.parent:
                    node.parent.children.remove(node)
                node.cost = new_node.cost + dist
                node.parent = new_node
                new_node.children.append(node)

    def add_new_node(self):
        while True:
            random_point = np.random.uniform(self.env.boundaries[0], self.env.boundaries[1], 2)
            nearest_node = self.find_nearest_node(random_point)
            new_point = self.steer(nearest_node, random_point)
            if new_point is not None:
                new_node = Node(new_point)
                new_node, success = self.find_proximal_node(new_node)
                if not success:
                    new_node.parent = nearest_node
                    nearest_node.children.append(new_node)
                self.nodes.append(new_node)
                self.rewire_nodes(new_node)
                return new_node

    def target_reached(self, node):
        return np.linalg.norm(node.get_position() - self.goal.get_position()) < self.search_radius

    def construct_path(self, goal_node):
        path = [goal_node.get_position()]
        current_node = goal_node
        while current_node.parent is not None:
            path.append(current_node.parent.get_position())
            current_node = current_node.parent
        path.reverse()
        return path

    def find_path(self):
        for _ in range(self.max_iterations):
            new_node = self.add_new_node()
            if self.target_reached(new_node):
                self.goal.parent = new_node
                self.nodes.append(self.goal)
                self.path = self.construct_path(self.goal)
                return self.path
        return None


# Usage example with ContinuousEnv
if __name__ == "__main__":
    from td3_copy.code.sim.gym_envirnoment_close_obstacles import ContinuousEnv

    # Initialize environment
    env = ContinuousEnv()

    # Run RRT* Smart
    rrt_star = RRTStarSmart(env, max_iterations=2000, search_radius=0.5, epsilon=1.0, rewiring_radius=1.5)
    optimal_path = rrt_star.find_path()

    # Visualize and wait for 'q' to close
    if optimal_path:
        print("Optimal Path (RRT*):", optimal_path)
        env.render(optimal_path=optimal_path)
    else:
        print("No path found.")


    # Use an event listener for 'q' to close the visualization window
    def on_key(event):
        if event.key == 'q':
            plt.close()


    # Connect the 'q' key event to the handler
    plt.gcf().canvas.mpl_connect('key_press_event', on_key)

    # Keep the window open
    print("Press 'q' to close the visualization window.")
    plt.show()
