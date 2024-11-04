import numpy as np
import random
import torch
from environment import PathPlanningEnv
from dqn import DQNAgent
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import time

# Set seeds for reproducibility
random.seed(30)
np.random.seed(30)

# Define best hyperparameters for DQN
best_hyperparams = {
    'learning_rate': 6.387267664933189e-05,
    'gamma': 0.8096888702479238,
    'epsilon_decay': 0.994029657582799,
    'min_epsilon': 0.03452338037756181,
    'step_penalty': -0.07751430378486499,
    'obstacle_penalty': -1.4332043037855677,
    'goal_reward': 995.665947512175,
    'episodes': 1185,
    'max_steps_per_episode': 4468
}

# RRT* Smart Algorithm
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

# Metric calculation functions
def calculate_path_length(path):
    return sum(np.linalg.norm(np.array(path[i]) - np.array(path[i - 1])) for i in range(1, len(path)))

def calculate_collisions(path, env):
    return sum(1 for point in path if tuple(point) in env.obstacles)

def calculate_clearance(path, env):
    return sum(min(np.linalg.norm(np.array(point) - np.array(obstacle)) for obstacle in env.obstacles) for point in path) / len(path)

def render_combined(env, dqn_path=None, rrt_path=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)

    # Draw grid
    for x in range(env.grid_size + 1):
        ax.axhline(x, lw=2, color='black', zorder=0)
        ax.axvline(x, lw=2, color='black', zorder=0)

    # Draw start and goal
    ax.add_patch(patches.Rectangle(env.start, 1, 1, edgecolor='blue', facecolor='lightblue', lw=2, label='Start'))
    ax.add_patch(patches.Rectangle(env.goal, 1, 1, edgecolor='green', facecolor='lightgreen', lw=2, label='Goal'))

    # Draw obstacles
    for obs in env.obstacles:
        ax.add_patch(patches.Rectangle(obs, 1, 1, edgecolor='red', facecolor='darkred', lw=2, label='Obstacle'))

    # Plot paths
    dqn_path_set = set(tuple(position) for position in dqn_path) if dqn_path else set()
    rrt_path_set = set(tuple(position) for position in rrt_path) if rrt_path else set()
    overlap_points = dqn_path_set & rrt_path_set
    only_dqn_points = dqn_path_set - overlap_points
    only_rrt_points = rrt_path_set - overlap_points

    for position in overlap_points:
        ax.plot(position[0] + 0.5, position[1] + 0.5, 'o', color='purple', markersize=8, label="Overlap (DQN & RRT*)")
    for position in only_dqn_points:
        ax.plot(position[0] + 0.5, position[1] + 0.5, 'bo', markersize=8, label="DQN Path")
    for position in only_rrt_points:
        ax.plot(position[0] + 0.5, position[1] + 0.5, 'ro', markersize=8, label="RRT* Smart Path")

    # Adding legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    plt.gca().invert_yaxis()
    plt.show(block=True)

if __name__ == "__main__":
    writer = SummaryWriter(log_dir="runs/DQN_RRT_comparison")

    env = PathPlanningEnv(grid_size=15, start=(0, 0), goal=(14, 14), random_obstacles=True)

    # RRT* Smart path
    rrt_star = RRTStarSmart(env, max_iterations=1000, search_radius=1, rewiring_radius=1)
    rrt_path = rrt_star.find_path()
    if rrt_path:
        rrt_path_length = calculate_path_length(rrt_path)
        rrt_collisions = calculate_collisions(rrt_path, env)
        rrt_clearance = calculate_clearance(rrt_path, env)
        rrt_steps = len(rrt_path)
    else:
        print("No RRT* path found.")
        rrt_path_length = rrt_collisions = rrt_clearance = rrt_steps = "N/A"

    # DQN path
    agent = DQNAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n,
                     learning_rate=best_hyperparams['learning_rate'], gamma=best_hyperparams['gamma'],
                     epsilon=1.0, epsilon_decay=best_hyperparams['epsilon_decay'],
                     min_epsilon=best_hyperparams['min_epsilon'],
                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    model_file = "dqn_model_random.pth"
    agent.load_model(model_file)

    state = env.reset()
    done = False
    dqn_path = []
    total_reward = 0

    for _ in range(best_hyperparams['max_steps_per_episode']):
        dqn_path.append(state)
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        if done:
            break

    dqn_path_length = calculate_path_length(dqn_path)
    dqn_collisions = calculate_collisions(dqn_path, env)
    dqn_clearance = calculate_clearance(dqn_path, env)
    dqn_steps = len(dqn_path)

    # Print the comparison table to console
    print("\nComparison of DQN and RRT* Smart Metrics:")
    print("=" * 50)
    print(f"{'Metric':<20} | {'DQN':<15} | {'RRT* Smart':<15}")
    print("=" * 50)
    print(f"{'Path Length':<20} | {dqn_path_length:<15.2f} | {rrt_path_length if rrt_path_length != 'N/A' else 'N/A':<15}")
    print(f"{'Collisions':<20} | {dqn_collisions:<15} | {rrt_collisions if rrt_collisions != 'N/A' else 'N/A':<15}")
    print(f"{'Clearance':<20} | {dqn_clearance:<15.2f} | {rrt_clearance if rrt_clearance != 'N/A' else 'N/A':<15}")
    print(f"{'Total Steps':<20} | {dqn_steps:<15} | {rrt_steps if rrt_steps != 'N/A' else 'N/A':<15}")
    print(f"{'Total Reward':<20} | {total_reward:<15.2f} | {'N/A':<15}")
    print("=" * 50)

    # Log metrics to TensorBoard as bar charts
    metrics = {
        "Path Length": [dqn_path_length, rrt_path_length],
        "Collisions": [dqn_collisions, rrt_collisions],
        "Clearance": [dqn_clearance, rrt_clearance],
        "Total Steps": [dqn_steps, rrt_steps],
        "Total Reward": [total_reward, 0]  # RRT* Smart does not have a reward metric
    }

    for metric_name, values in metrics.items():
        fig, ax = plt.subplots()
        ax.bar(["DQN", "RRT* Smart"], values, width=0.3, color=['#1f77b4', '#ff7f0e'])
        ax.set_title(f"{metric_name} Comparison")
        ax.set_ylabel(metric_name)
        writer.add_figure(f"{metric_name}_Comparison", fig)
        plt.close(fig)

    # Render combined paths
    if rrt_path:
        render_combined(env, dqn_path=dqn_path, rrt_path=rrt_path)

    writer.close()
    print("Metrics logged to TensorBoard as bar charts.")
