import time
import gym
import numpy as np
import optuna
from environment import PathPlanningEnv
from dqn import DQNAgent
import torch
import os
import matplotlib.pyplot as plt


# Define the Optuna objective function
def objective(trial):
    # Define the hyperparameters to be tuned by Optuna
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.8, 0.999)
    epsilon_decay = trial.suggest_float('epsilon_decay', 0.99, 0.99999)
    min_epsilon = trial.suggest_float('min_epsilon', 0.01, 0.15)
    step_penalty = trial.suggest_float('step_penalty', -1, 0)
    obstacle_penalty = trial.suggest_float('obstacle_penalty', -10, -1)
    goal_reward = trial.suggest_float('goal_reward', 500, 1000)
    episodes = trial.suggest_int('episodes', 1000, 7000)
    max_steps_per_episode = trial.suggest_int('max_steps_per_episode', 500, 5000)

    # Initialize the environment
    env = PathPlanningEnv(grid_size=15, start=(0, 0), goal=(14, 14),
                          step_penalty=step_penalty,
                          obstacle_penalty=obstacle_penalty,
                          goal_reward=goal_reward, random_obstacles=True, seed=int(time.time()))

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Specify the custom log directory
    log_dir = '/media/nouha/EXTERNAL_USB/pfe'

    agent = DQNAgent(state_dim, action_dim, learning_rate=learning_rate, gamma=gamma,
                     epsilon=1.0, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon,
                     device=device, log_dir=log_dir)

    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < max_steps_per_episode:
            steps += 1
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.replay(episode)
            state = next_state
            total_reward += reward

        total_rewards.append(total_reward)

        if episode % 10 == 0:
            agent.update_target_model()

    # Return the average reward as the metric to optimize
    avg_reward = np.mean(total_rewards)
    return avg_reward


# Create the Optuna study and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Print the best hyperparameters found by Optuna
best_hyperparams = study.best_params
print("Best hyperparameters found:", best_hyperparams)

# Extract the best number of episodes and max_steps_per_episode
episodes = best_hyperparams.pop('episodes')
max_steps_per_episode = best_hyperparams.pop('max_steps_per_episode')

# Initialize the environment with the best hyperparameters found by Optuna
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

# Specify the custom log directory
log_dir = '/media/nouha/1616C64516C62619/Users/essid/Documents/tensorboard_logs'

agent = DQNAgent(state_dim, action_dim, learning_rate=best_hyperparams['learning_rate'],
                 gamma=best_hyperparams['gamma'],
                 epsilon=1.0, epsilon_decay=best_hyperparams['epsilon_decay'],
                 min_epsilon=best_hyperparams['min_epsilon'], device=device,
                 log_dir=log_dir)

# Initialize path_history to avoid NameError
path_history = []

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

# Train the model only if no model was loaded
if not model_loaded:
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        path = []  # Initialize the path for this episode

        while not done and steps < max_steps_per_episode:
            steps += 1
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.replay(episode)
            state = next_state
            total_reward += reward

            path.append(state)  # Add the current state to the path

        path_history.append(path)  # Append the entire path for this episode
        print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}, Steps: {steps}, Epsilon: {agent.epsilon:.4f}")

        if episode % 10 == 0:
            agent.update_target_model()

    # Save the trained model
    agent.save_model(model_file)
    print(f"Model saved as {model_file}.")

# If the model was loaded, you can evaluate or continue fine-tuning it
else:
    print("Model loaded. You can now evaluate or continue fine-tuning the model.")
    # Optional: Evaluate the model, fine-tune further, or simply run a few episodes to see its performance
    state = env.reset()
    done = False
    path = []
    total_reward = 0
    steps = 0

    while not done and steps < max_steps_per_episode:
        steps += 1
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        path.append(state)
        total_reward += reward

    # Print out the results of the evaluation
    print(f"Evaluation completed: Reward: {total_reward}, Steps: {steps}")

    # Render the final path on the grid using blue points
    env.render(path=path)

# Plot the learning progress if training was done
if not model_loaded:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot([episode for episode in range(episodes)], [total_reward for _ in range(episodes)])
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    plt.subplot(1, 2, 2)
    plt.plot([episode for episode in range(episodes)], [agent.epsilon for _ in range(episodes)])
    plt.title("Epsilon Decay")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")

    plt.show
