import numpy as np
import random
from environment import PathPlanningEnv
import matplotlib.pyplot as plt
import matplotlib.patches as patches

random.seed(30)  # Set the seed for Python's random module
np.random.seed(30)  # Set the seed for NumPy's random module

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
    def __init__(self, env, max_iterations=1000, search_radius=1, rewiring_radius=1):
        self.env = env
        self.start = Node(self.env.start, start_node=True)
        self.goal = Node(self.env.goal, target_node=True)
        self.nodes = [self.start]
        self.max_iterations = max_iterations
        self.search_radius = search_radius
        self.rewiring_radius = rewiring_radius
        self.path = []

    def is_collision_free(self, point):
        return (point[0], point[1]) not in self.env.obstacles

    def find_nearest_node(self, point):
        distances = [np.sum(np.abs(node.get_position() - point)) for node in self.nodes]
        nearest_index = np.argmin(distances)
        return self.nodes[nearest_index]

    def steer(self, from_node, to_point):
        direction = to_point - from_node.get_position()
        distance = np.sum(np.abs(direction))

        if distance > self.search_radius:
            direction = direction / distance * self.search_radius

        potential_points = [
            from_node.get_position() + np.array([0, 1]),
            from_node.get_position() + np.array([0, -1]),
            from_node.get_position() + np.array([1, 0]),
            from_node.get_position() + np.array([-1, 0])
        ]

        for point in potential_points:
            if np.all(point == to_point) and self.is_collision_free(point):
                return point
        return None

    def find_proximal_node(self, new_node):
        proximal_node = None
        for node in self.nodes:
            dist = np.sum(np.abs(new_node.get_position() - node.get_position()))
            if dist < self.rewiring_radius and node.cost + dist < new_node.cost:
                proximal_node = node
                new_node.cost = node.cost + dist
                new_node.parent = proximal_node
        if proximal_node:
            proximal_node.children.append(new_node)
        return new_node, proximal_node is not None

    def rewire_nodes(self, new_node):
        for node in self.nodes:
            dist = np.sum(np.abs(new_node.get_position() - node.get_position()))
            if dist < self.rewiring_radius and new_node.cost + dist < node.cost:
                if node.parent:
                    node.parent.children.remove(node)
                node.cost = new_node.cost + dist
                node.parent = new_node
                new_node.children.append(node)

    def add_new_node(self):
        while True:
            random_point = np.array([random.randint(0, self.env.grid_size - 1), random.randint(0, self.env.grid_size - 1)])
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
        return np.array_equal(node.get_position(), self.goal.get_position())

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


# Updated render function
def render(env, path=None, optimal_path=None, path_label="DQN Path", optimal_path_label="RRT* Path"):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)

    # Draw grid
    for x in range(env.grid_size + 1):
        ax.axhline(x, lw=2, color='black', zorder=0)
        ax.axvline(x, lw=2, color='black', zorder=0)

    # Draw start
    ax.add_patch(patches.Rectangle(env.start, 1, 1, edgecolor='blue', facecolor='lightblue', lw=2, label='Start'))

    # Draw goal
    ax.add_patch(patches.Rectangle(env.goal, 1, 1, edgecolor='green', facecolor='lightgreen', lw=2, label='Goal'))

    # Draw obstacles
    for obs in env.obstacles:
        ax.add_patch(patches.Rectangle(obs, 1, 1, edgecolor='red', facecolor='darkred', lw=2, label='Obstacle'))

    # Convert paths to sets of tuples for easier comparison
    dqn_path_set = set(tuple(position) for position in path) if path else set()
    rrt_path_set = set(tuple(position) for position in optimal_path) if optimal_path else set()

    # Find overlapping points between DQN and RRT* paths
    overlap_points = dqn_path_set & rrt_path_set
    only_dqn_points = dqn_path_set - overlap_points
    only_rrt_points = rrt_path_set - overlap_points

    # Plot overlapping points in purple
    for position in overlap_points:
        ax.plot(position[0] + 0.5, position[1] + 0.5, 'o', color='purple', markersize=8,
                label="Overlap (DQN & RRT*)")

    # Plot DQN-only path points in blue
    for position in only_dqn_points:
        ax.plot(position[0] + 0.5, position[1] + 0.5, 'bo', markersize=8, label=path_label)

    # Plot RRT*-only path points in red
    for position in only_rrt_points:
        ax.plot(position[0] + 0.5, position[1] + 0.5, 'ro', markersize=8, label=optimal_path_label)

    # Adding legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    # Set grid labels
    ax.set_xticks(np.arange(0.5, env.grid_size, 1))
    ax.set_yticks(np.arange(0.5, env.grid_size, 1))
    ax.set_xticklabels(np.arange(1, env.grid_size + 1))
    ax.set_yticklabels(np.arange(1, env.grid_size + 1))

    ax.grid(False)
    plt.gca().invert_yaxis()  # Invert Y-axis to match the grid indexing with matrix indexing
    plt.show(block=True)  # Keep the window open until manually closed


# Usage example with PathPlanningEnv
if __name__ == "__main__":
    env = PathPlanningEnv(grid_size=15, start=(0, 0), goal=(14, 14), random_obstacles=True)

    # Run RRT* Smart in the environment
    rrt_star = RRTStarSmart(env, max_iterations=1000, search_radius=1, rewiring_radius=1)
    optimal_path = rrt_star.find_path()

    if optimal_path:
        print("Optimal Path (RRT* Smart):", optimal_path)
        render(env, path=optimal_path, path_label="RRT* Smart Path")
    else:
        print("No path found.")
