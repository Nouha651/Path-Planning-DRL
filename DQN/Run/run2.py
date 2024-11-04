import time
import gym
import numpy as np
from environment1 import PathPlanningEnv
from dqn import DQNAgent
import torch
import os
import matplotlib.pyplot as plt

# Best hyperparameters found by Optuna
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

# Initialize the environment with the best hyperparameters and a fixed seed
use_random_environment = True

if use_random_environment:
    env = PathPlanningEnv(grid_size=15, start=(0, 0), goal=(14, 14),
                          step_penalty=best_hyperparams['step_penalty'],
                          obstacle_penalty=best_hyperparams['obstacle_penalty'],
                          goal_reward=best_hyperparams['goal_reward'], random_obstacles=True, seed=int(time.time()))
else:
    fixed_obstacles = [
        (0, 3), (1, 4), (1, 5),  # Horizontal block near the top left
        (3, 3), (3, 4), (3, 5),  # Small block in the center-left
        (6, 6), (6, 7), (6, 8),  # Horizontal block in the middle
        (10, 3), (10, 5),  # Horizontal block in the bottom-left
        (12, 8), (13, 8), (14, 8),  # Vertical block near the bottom right
        (5, 10), (6, 10), (8, 10),  # Vertical block in the middle-right
        (11, 13), (12, 13)  # Horizontal block near the bottom right
    ]
    env = PathPlanningEnv(grid_size=15, obstacles=fixed_obstacles, start=(0, 0), goal=(14, 14),
                          step_penalty=best_hyperparams['step_penalty'],
                          obstacle_penalty=best_hyperparams['obstacle_penalty'],
                          goal_reward=best_hyperparams['goal_reward'])

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

agent = DQNAgent(state_dim, action_dim, learning_rate=best_hyperparams['learning_rate'],
                 gamma=best_hyperparams['gamma'],
                 epsilon=1.0, epsilon_decay=best_hyperparams['epsilon_decay'],
                 min_epsilon=best_hyperparams['min_epsilon'], device=device)

# Define the number of episodes and maximum steps per episode
episodes = best_hyperparams['episodes']
max_steps_per_episode = best_hyperparams['max_steps_per_episode']

# Set model file name based on the environment type
model_file = "dqn_model_random.pth" if use_random_environment else "dqn_model_fixed.pth"
model_loaded = False  # Flag to check if the model was loaded

# Attempt to load the existing model, if it exists
if os.path.exists(model_file):
    print(f"Loading existing model from {model_file}...")
    agent.load_model(model_file)
    model_loaded = True
else:
    print(f"No saved model found, training a new one for {model_file}...")


def run_episode(render=False):
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0
    path = [state]  # Start by adding the initial state to the path

    while not done and steps < max_steps_per_episode:
        steps += 1
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        path.append(state)  # Add each state to the path

    return total_reward, path, steps


# Perform a single evaluation
total_reward, path, steps = run_episode(render=False)

# Print out the results of the evaluation
print(f"Evaluation completed: Reward: {total_reward}, Steps: {steps}")

# Render and visualize only if the reward exceeds 980
if total_reward > 950:
    print(f"Successful run: Reward = {total_reward}")
    env.render(path=path)  # Ensure the full path is passed for rendering
else:
    print(f"Reward {total_reward} did not meet the threshold. Skipping visualization.")
